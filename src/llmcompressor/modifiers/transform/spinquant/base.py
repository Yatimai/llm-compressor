import math
from enum import Enum
from typing import Iterable, List, Literal, Optional

import torch
import torch.nn as nn
from compressed_tensors import match_modules_set, match_named_modules
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import TorchDtype, get_head_dim
from datasets import load_dataset
from loguru import logger
from pydantic import Field, field_validator
from torch.utils._pytree import tree_leaves
from transformers import AutoTokenizer, PreTrainedModel

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.modifiers import Modifier
from llmcompressor.typing import NamedModules
from llmcompressor.utils import untie_word_embeddings

from .cayley_sgd import CayleySGD
from .factories import LearnableHadamardFactory
from .mappings import SpinQuantMapping, infer_mapping_from_model
from .norm_mappings import NormMapping, infer_norm_mapping_from_model
from .rtn_fake_quant import add_rtn_fake_quant, remove_rtn_fake_quant


class SpinquantRotation(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class SpinQuantModifier(Modifier, use_enum_values=True):
    """
    Implements the transforms according to "SpinQuant: LLM quantization
    with learned rotations" (https://arxiv.org/abs/2405.16406)

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achieved through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    The SpinQuant authors describe four different rotations which can be applied to a
    model. R1 and R2 are "offline" rotations, meaning that they can be fused into
    existing weights and therefore do not induce runtime cost. R3 and R4 are "online"
    rotations, meaning that they require additional computation at runtime.

    When ``learnable=True``, the R1 and R2 rotation matrices are trained via
    Cayley SGD on the Stiefel manifold (Li et al. 2020, arXiv:2002.01113)
    against an RTN-quantized loss landscape on WikiText-2 calibration data.
    R3 and R4 remain as fixed Hadamard transforms matching the SpinQuant
    paper design. The learnable path is an Apache-2.0 reimplementation based
    on the paper's description, independent of the original reference code.

    Lifecycle:

    - on_initialize
        - infer SpinQuantMappings & NormMappings
        - as needed, create transform schemes for R1, R2, R3, & R4
    - on_start
        - normalize embeddings
        - fuse norm layers into subsequent Linear layers
        - apply TransformConfig
            - fuse transforms into weights for mergeable transforms
            - add hooks for online transforms
    - on sequential epoch end
    - on_end
    - on_finalize

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-hadamard"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application
    :param learnable: if True, attach gradients to transform weights for training
    :param precision: Precision at which all transforms should be applied. This applies
        to both weight fusing and online rotations
    :param transform_block_size: Block size to use for rotation matrices. The model's
        hidden_size, intermediate_size, and head_dim must be evenly divisible by
        transform_block_size. Layers will be transformed by a block-diagonal matrix
        where each block is a matrix of this size.
        If None is provided, the targeted weight dimension is used: hidden_size for
        R1, head_dim for R2, head_dim for R3, and intermediate_size for R4 on
        down_proj. For R4 on Llama-class models a value of 128 is typical (matches
        the original SpinQuant fast Hadamard kernel block size).
    :param mappings: Specifies layers within a model to target for transforms.
        A mapping will be inferred if None is provided
    :param norm_mappings: Specifies layers within a model to target for norm fusing.
        A mapping will be inferred if None is provided
    :param transform_config: Optional transform config for overriding provided arguments
    :param cayley_num_samples: Number of calibration samples from WikiText-2
        train used for Cayley SGD training. Only consumed when ``learnable=True``.
        Defaults to 800, as used in the SpinQuant setting.
    :param cayley_batch_size: Batch size for forward passes during Cayley
        training. Defaults to 1. On single-GPU setups, combine with
        ``cayley_gradient_accumulation_steps`` to reach the desired effective
        batch size.
    :param cayley_gradient_accumulation_steps: Number of forward-backward
        passes accumulated before each optimizer step. Defaults to 8.
        Combined with the default ``cayley_batch_size=1`` and
        ``cayley_num_samples=800``, yields 100 optimizer steps.
    :param cayley_lr: Initial learning rate for Cayley SGD, decayed to 0 via
        a cosine schedule over the optimizer steps. Defaults to 1.5.
    """

    rotations: List[SpinquantRotation] = Field(default_factory=lambda: ["R1", "R2"])
    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="hadamard"
    )
    randomize: bool = Field(default=False)
    learnable: bool = Field(default=False)
    precision: TorchDtype = Field(default=torch.float64)
    transform_block_size: Optional[int] = Field(default=None)

    # Cayley SGD hyperparameters (only used when learnable=True)
    cayley_num_samples: int = Field(default=800)
    cayley_batch_size: int = Field(default=1)
    cayley_gradient_accumulation_steps: int = Field(default=8)
    cayley_lr: float = Field(default=1.5)

    # norm mappings separate from spinquant mappings to allow users to
    # override spinquant mappings with transform_config without overriding norms
    mappings: Optional[SpinQuantMapping] = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    norm_mappings: Optional[List[NormMapping]] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    @field_validator("randomize", mode="before")
    def validate_randomize_not_implemented(cls, value):
        if value:
            raise NotImplementedError("randomize is not supported as of now")
        return value

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.mappings = infer_mapping_from_model(state.model)
        self.norm_mappings = infer_norm_mapping_from_model(state.model)
        head_dim = get_head_dim(state.model.config)

        config_groups = {}
        if SpinquantRotation.R1 in self.rotations:
            config_groups["R1"] = self._create_r1_scheme()

        if SpinquantRotation.R2 in self.rotations:
            config_groups["R2"] = self._create_r2_scheme(head_dim)

        if SpinquantRotation.R3 in self.rotations:
            config_groups["R3"] = self._create_r3_scheme(head_dim)

        if SpinquantRotation.R4 in self.rotations:
            config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    @torch.no_grad()
    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        model = state.model

        # untie embeddings to avoid unintended effects of `_center_embeddings`
        untie_word_embeddings(model)

        # needs to happen after the model has been hooked to execute on the GPU
        # otherwise we're applying weight transforms on CPU
        self._center_embeddings(model)
        self._fuse_norms(model)

        # Apply R4 (fixed block-diagonal Hadamard, head_dim=128) BEFORE the
        # learnable transform_config so R4 fuses into down_proj raw weight
        # while the module is still parametrize-free. Applying R4 AFTER
        # R1's parametrization would route `update_offload_parameter`
        # through R1's `right_inverse`, causing R4 to be applied twice in
        # the forward chain. Follows the SpinQuant paper's canonical
        # ordering (Liu et al. 2024, arXiv:2405.16406): norm fusion ->
        # fixed online Hadamards -> learnable rotation training, so
        # R1/R2 are optimized against the R4-quantized landscape.
        # Only triggered when R4 is not explicitly in ``self.rotations``;
        # users who pass ``rotations=["R1","R2","R4"]`` opt into the
        # configurable path via ``transform_block_size``.
        if self.learnable and "R4" not in self.rotations:
            self._apply_r4(model)

        if self.learnable:
            # apply fixed schemes (requires_grad=False) first via the upstream
            # path so R3/R4 online Hadamards bake into plain Linear weights
            # before the learnable schemes parametrize the same modules
            # (compressed_tensors rejects ParametrizedLinear in apply_transform_weight)
            non_learnable = {
                name: scheme
                for name, scheme in self.transform_config.config_groups.items()
                if not scheme.requires_grad
            }
            if non_learnable:
                apply_transform_config(
                    model, TransformConfig(config_groups=non_learnable)
                )
                # apply_transform_config overwrites model.transform_config
                # with the sub-config passed in. Restore the full config so
                # config.json serializes both learnable (R1/R2) and fixed
                # (R3/R4) schemes for downstream reconstruction.
                setattr(model, "transform_config", self.transform_config)
            # apply learnable schemes (R1 shared, R2 per-layer) via the
            # custom factory so Cayley SGD can optimize the rotation parameters
            self._apply_learnable_transform_config(model)
            self._tie_r2_per_layer(model)
            self._run_cayley_training(model)
        else:
            apply_transform_config(model, self.transform_config)

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            pass

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        return True

    def _get_targets(self, model: torch.nn.Module) -> NamedModules:
        return [
            (name, module)
            for scheme in self.transform_config.config_groups.values()
            for arg in scheme.apply
            for name, module in match_named_modules(model, arg.targets, arg.ignore)
        ]

    def _center_embeddings(self, model: PreTrainedModel):
        for _, embedding in match_named_modules(
            model, [self.mappings.embedding], warn_on_fail=True
        ):
            center_embeddings(embedding)

    def _fuse_norms(self, model: PreTrainedModel):
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                # match_modules_set returns a list of lists
                assert len(norm) == 1
                fuse_norm_linears(norm[0], tree_leaves(linears))

    def _create_r1_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[
                        self.mappings.embedding,
                        self.mappings.attn_o,
                        *self.mappings.mlp_out,
                    ],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        self.mappings.attn_q,
                        self.mappings.attn_k,
                        self.mappings.attn_v,
                        *self.mappings.mlp_in,
                        self.mappings.lm_head,
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r2_scheme(self, head_dim: int) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=head_dim,
            apply=[
                TransformArgs(targets=[self.mappings.attn_v], location="weight_output"),
                TransformArgs(
                    targets=[self.mappings.attn_o],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r3_scheme(self, head_dim: int) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            # R3 is online at runtime and must stay Hadamard for kernel efficiency
            requires_grad=False,
            precision=self.precision,
            head_dim=head_dim,
            apply=[
                TransformArgs(
                    targets=[self.mappings.attn],
                    location="q_attn",
                ),
                TransformArgs(
                    targets=[self.mappings.attn],
                    location="k_cache",
                ),
            ],
        )

    def _apply_learnable_transform_config(self, model: nn.Module) -> None:
        """
        Apply the learnable Hadamard rotation schemes via
        :class:`LearnableHadamardFactory` instead of the upstream
        :func:`apply_transform_config`. This factory accepts Linear subclasses
        (so stacked parametrizations compose) and creates per-module weights
        for R2 (so each attention layer holds its own learnable Parameter).

        Non-learnable schemes are skipped here and applied separately via the
        upstream path in :py:meth:`on_start` before the learnable rotations to
        ensure plain Linear weights are available when ``apply_transform_weight``
        runs its strict type check.
        """
        for name, scheme in self.transform_config.config_groups.items():
            if not scheme.requires_grad:
                continue
            if scheme.type == "hadamard":
                factory = LearnableHadamardFactory(name=name, scheme=scheme)
            else:
                factory = TransformFactory.from_scheme(scheme, name=name)
            factory.apply_to_model(model)

    def _apply_r4(self, model: PreTrainedModel) -> None:
        """
        Apply R4 (fixed block-diagonal Hadamard on ``mlp_out``, head_dim=128)
        before the learnable transform_config is attached to the model.

        R4 stays ``requires_grad=False`` and is not part of the learnable
        optimization set. ``head_dim`` is hardcoded to 128 independently of
        ``self.transform_block_size`` (which controls R1/R2/R3 sizing) because
        R4 uses the SpinQuant recipe where 128 matches the fast Hadamard
        kernel block size optimal for Llama-class models.

        Ordering constraint: R4 must fuse into ``down_proj.weight`` while the
        module is still parametrize-free to avoid an ambiguous interaction
        with compressed_tensors' parametrize-aware weight update path. See
        ``INTEGRATION_WORKAROUNDS.md`` section W8 for the full technical
        explanation.

        Only triggered in :py:meth:`on_start` when R4 is not explicitly listed
        in ``self.rotations``. Users who pass ``rotations=["R1","R2","R4"]``
        opt into the configurable path via ``transform_block_size`` in the
        standard :py:meth:`_create_r4_scheme` flow.
        """
        r4_scheme = TransformScheme(
            type=self.transform_type,
            requires_grad=False,
            precision=self.precision,
            head_dim=128,
            apply=[
                TransformArgs(targets=[*self.mappings.mlp_out], location="input"),
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )
        apply_transform_config(model, TransformConfig(config_groups={"R4": r4_scheme}))

    def _create_r4_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            # R4 is online at runtime and must stay Hadamard for kernel efficiency
            requires_grad=False,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="input",
                ),
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _tie_r2_per_layer(self, model: nn.Module) -> None:
        """
        Tie R2 instances within each attention layer: force o_proj's R2 to
        share the same Parameter as v_proj's R2. Required by SpinQuant math,
        since R2 must cancel between v_proj (output side) and o_proj (input
        side, inverse) for the attention output to remain mathematically valid.

        The per-module ``LearnableHadamardFactory.create_transform`` creates
        one fresh Parameter per Linear, which is correct *per layer* but
        incorrect *within a layer* (v_proj and o_proj must share). This
        post-pass restores the per-layer tying.
        """
        head_dim = get_head_dim(model.config)

        def _find_r2(linear: nn.Module) -> Optional[nn.Module]:
            plists = getattr(linear, "parametrizations", None)
            if plists is None:
                return None
            for plist in plists.values():
                for transform in plist:
                    weight = getattr(transform, "weight", None)
                    if weight is not None and weight.shape == (head_dim, head_dim):
                        return transform
            return None

        for v_matches, o_matches in match_modules_set(
            model, (self.mappings.attn_v, self.mappings.attn_o)
        ):
            # match_modules_set groups one v_proj with its paired o_proj per layer
            assert len(v_matches) == 1 and len(o_matches) == 1
            v_proj, o_proj = v_matches[0], o_matches[0]
            v_r2 = _find_r2(v_proj)
            if v_r2 is None:
                continue
            o_r2 = _find_r2(o_proj)
            if o_r2 is None or id(o_r2.weight) == id(v_r2.weight):
                continue
            # replace o_proj R2's Parameter with v_proj R2's Parameter
            del o_r2._parameters["weight"]
            o_r2.weight = v_r2.weight

    def _collect_rotation_params(self, model: nn.Module) -> List[torch.nn.Parameter]:
        """
        Return the unique learnable rotation matrices added by
        `apply_transform_config`. Modules sharing a matrix (same shape group)
        appear only once.
        """
        # deduplicate by id(p): R1 is a single Parameter shared across all
        # modules where it is applied, and must be optimized only once
        seen: set = set()
        params: List[torch.nn.Parameter] = []
        for module in model.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None:
                continue
            for plist in parametrizations.values():
                for transform in plist:
                    for p in transform.parameters():
                        if not p.requires_grad:
                            continue
                        if id(p) in seen:
                            continue
                        seen.add(id(p))
                        params.append(p)
        return params

    def _run_cayley_training(self, model: PreTrainedModel) -> None:
        """
        Optimize the learnable rotations (R1, R2) with Cayley SGD on the Stiefel
        manifold. RTN fake-quantization is applied via a stacked parametrization
        so the rotation sees the quantization loss during backward.
        """
        # _tie_r2_per_layer was already called in on_start
        rotation_params = self._collect_rotation_params(model)
        rotation_ids = {id(p) for p in rotation_params}
        for p in model.parameters():
            if id(p) not in rotation_ids:
                p.requires_grad_(False)

        logger.info(
            "cayley: {} learnable rotation parameters, shapes: {}",
            len(rotation_params),
            [tuple(p.shape) for p in rotation_params[:5]],
        )

        num_samples = self.cayley_num_samples
        batch_size = self.cayley_batch_size
        accum_steps = self.cayley_gradient_accumulation_steps
        if num_samples % (batch_size * accum_steps) != 0:
            raise ValueError(
                f"cayley_num_samples {num_samples} must be divisible by "
                f"cayley_batch_size * cayley_gradient_accumulation_steps "
                f"= {batch_size} * {accum_steps}"
            )
        # 800 samples x batch=1 with gradient accumulation 8 gives 100
        # optimizer steps
        total_optimizer_steps = num_samples // (batch_size * accum_steps)
        seq_len = 2048
        base_lr = self.cayley_lr

        # calibration data: WikiText-2 train, non-overlapping 2048-token chunks
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(x["text"] for x in raw if x["text"].strip())
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]

        total_tokens_needed = num_samples * seq_len
        if tokens.numel() < total_tokens_needed:
            raise ValueError(
                f"WikiText-2 train has {tokens.numel()} tokens, need "
                f"{total_tokens_needed} for {num_samples} samples of {seq_len}"
            )

        device = next(model.parameters()).device
        batches = []
        for i in range(num_samples // batch_size):
            start = i * batch_size * seq_len
            batch = tokens[start : start + batch_size * seq_len].view(
                batch_size, seq_len
            )
            batches.append(batch.to(device))

        patched = add_rtn_fake_quant(model, num_bits=4, group_size=128)
        optimizer = CayleySGD(rotation_params, lr=base_lr, momentum=0.0)

        # gradient checkpointing reduces activation memory for the training loop
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        accum_loss = 0.0
        try:
            with torch.enable_grad():
                for i, input_ids in enumerate(batches):
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    # scale loss so accumulated gradient is the average, matching DDP
                    loss = outputs.loss / accum_steps
                    loss.backward()
                    accum_loss += loss.item() * accum_steps

                    if (i + 1) % accum_steps == 0:
                        # cosine schedule over optimizer steps (not forward passes)
                        current_lr = (
                            base_lr
                            * 0.5
                            * (
                                1.0
                                + math.cos(math.pi * step_count / total_optimizer_steps)
                            )
                        )
                        optimizer.param_groups[0]["lr"] = current_lr
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        mean_loss = accum_loss / accum_steps
                        accum_loss = 0.0

                        if step_count < 20 or step_count % 10 == 0:
                            logger.info(
                                "cayley step {}/{} lr: {:.4f} loss: {:.4f}",
                                step_count,
                                total_optimizer_steps,
                                current_lr,
                                mean_loss,
                            )

                        if step_count == 0:
                            logger.info(
                                "VRAM after step 0: {:.1f} GB",
                                torch.cuda.max_memory_allocated() / 1e9,
                            )

                        step_count += 1
        finally:
            remove_rtn_fake_quant(patched)
            model.gradient_checkpointing_disable()

        self._collect_rotation_matrices(model)

    def _collect_rotation_matrices(self, model: nn.Module) -> None:
        """Collect learnable rotation matrices into ``self.learned_rotations``."""
        rotations: dict = {}
        seen: set = set()
        for name, p in model.named_parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                rotations[name] = p.detach().cpu()

        # bypass pydantic: ``SpinQuantModifier`` is a Modifier (Pydantic model)
        # and does not declare ``learned_rotations`` as a field. Using
        # ``object.__setattr__`` attaches it as a plain attribute without
        # triggering field validation. See INTEGRATION_WORKAROUNDS.md for
        # the full list of bypasses in the learnable path.
        object.__setattr__(self, "learned_rotations", rotations)
        logger.info(
            "cayley: collected {} rotation matrices",
            len(rotations),
        )
