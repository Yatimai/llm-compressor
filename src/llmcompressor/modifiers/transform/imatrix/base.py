import math
from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.utils import match_named_modules
from loguru import logger
from pydantic import Field
from torch.nn import Module

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier

__all__ = ["IMatrixGatherer"]

IMATRIX_PRECISION = torch.float32


class IMatrixGatherer(Modifier):
    """
    Collects importance-weighted activation statistics (E[x²])
    for each targeted module via forward pre-hooks.

    Stores ``module._imatrix_importance`` as a 1-D float32 tensor
    of shape ``(in_features,)`` on each target module.  Does **not**
    quantize or modify weights in any way.

    The downstream ``imatrix_mse`` observer reads this attribute during
    its grid search to weight quantization error by channel importance.

    Statistics are kept on GPU during calibration for speed, then
    offloaded to CPU at CALIBRATION_EPOCH_END to free GPU memory
    before quantization begins.

    Example recipe::

        recipe:
          - IMatrixGatherer:
              ignore: ["lm_head"]
          - QuantizationModifier:
              config_groups:
                group_0:
                  targets: ["Linear"]
                  weights:
                    observer: imatrix_mse

    Or composed with AWQ and GPTQ::

        recipe:
          - AWQModifier(...)
          - IMatrixGatherer:
              ignore: ["lm_head"]
          - GPTQModifier:
              config_groups:
                group_0:
                  targets: ["Linear"]
                  weights:
                    observer: imatrix_mse

    .. note::
        Auto-prepend (inserting the gatherer automatically when
        ``imatrix_mse`` is detected in a recipe) is planned for a
        follow-up PR.

    .. note::
        Unlike AWQModifier, this gatherer does not use IntermediatesCache
        because it only stores a single accumulated 1-D tensor per layer
        (not full batch activations).  A simple CPU offload at
        CALIBRATION_EPOCH_END is sufficient.

    :param ignore: layer name patterns to skip (default: ``["lm_head"]``)
    :param targets: module types to instrument (default: ``["Linear"]``)
    """

    ignore: Union[str, List[str]] = Field(
        default_factory=lambda: ["lm_head"],
    )
    targets: Union[str, List[str]] = Field(
        default_factory=lambda: ["Linear"],
    )

    # -- internal state (excluded from serialisation) --
    _target_names: Optional[List[str]] = None
    _sums: Optional[Dict[Module, torch.Tensor]] = None
    _counts: Optional[Dict[Module, int]] = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.end and self.end != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied "
                f"during one-shot.  Expected end to be None or "
                f"-1, got {self.end}"
            )
        if self.start and self.start != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied "
                f"during one-shot.  Expected start to be None "
                f"or -1, got {self.start}"
            )

        self._resolve_targets(state.model)
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        self._register_accumulation_hooks(state.model)

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._compute_and_attach(state.model)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._compute_and_attach(state.model, offload_to_cpu=True)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        # Clean up importance tensors from modules
        for _, module in match_named_modules(state.model, self.targets, self.ignore):
            if hasattr(module, "_imatrix_importance"):
                del module._imatrix_importance

        self._sums = None
        self._counts = None
        self._target_names = None
        return True

    # ------------------------------------------------------------------ #
    #  Target resolution
    # ------------------------------------------------------------------ #

    def _resolve_targets(self, model: Module):
        """Identify target modules using compressed_tensors matching."""
        self._target_names = []
        self._sums = {}
        self._counts = {}

        for name, module in match_named_modules(model, self.targets, self.ignore):
            if not hasattr(module, "in_features"):
                continue

            self._target_names.append(name)
            self._sums[module] = torch.zeros(module.in_features, dtype=torch.float32)
            self._counts[module] = 0

        logger.info(f"IMatrixGatherer: targeting {len(self._target_names)}" f" modules")

    # ------------------------------------------------------------------ #
    #  Hook registration
    # ------------------------------------------------------------------ #

    def _register_accumulation_hooks(self, model: Module):
        """Attach a forward-pre hook to every target module."""

        def _hook(module: Module, args):
            x = args[0] if not isinstance(args, torch.Tensor) else args
            if isinstance(x, tuple):
                x = x[0]
            if not isinstance(x, torch.Tensor):
                return

            # Per-token accumulation
            x_f = x.detach().to(IMATRIX_PRECISION)
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            device = self._sums[module].device
            if device != token_sum.device:
                self._sums[module] = self._sums[module].to(token_sum.device)

            self._sums[module].add_(token_sum)
            self._counts[module] += n_tokens

        for name, module in match_named_modules(model, self.targets, self.ignore):
            if module in self._sums:
                self.register_hook(module, _hook, "forward_pre")

    # ------------------------------------------------------------------ #
    #  Compute & attach
    # ------------------------------------------------------------------ #

    def _compute_and_attach(self, model: Module, offload_to_cpu: bool = False):
        """
        Compute E[x²] and store on each module.

        :param model: model whose modules receive importance data
        :param offload_to_cpu: if True, move importance tensors to CPU
            after attaching.  Set at CALIBRATION_EPOCH_END to free
            GPU memory before quantization.
        """
        attached = 0
        for name, module in match_named_modules(model, self.targets, self.ignore):
            if module not in self._sums:
                continue

            count = self._counts[module]
            if count == 0:
                continue

            importance = self._sums[module] / count

            if offload_to_cpu:
                importance = importance.to("cpu")
                # also free the accumulator
                del self._sums[module]

            module._imatrix_importance = importance

            attached += 1
            logger.debug(
                f"iMatrix {name}: "
                f"mean={importance.mean():.4f}, "
                f"max={importance.max():.4f}, "
                f"ratio="
                f"{importance.max() / (importance.mean() + 1e-10):.1f}"
            )

        logger.info(
            f"IMatrixGatherer: attached importance to "
            f"{attached} modules" + (" (offloaded to CPU)" if offload_to_cpu else "")
        )
