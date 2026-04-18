# INTEGRATION_WORKAROUNDS: SpinQuant Cayley learnable path

This document lists the eight workarounds required to make learnable
SpinQuant rotations (`CayleySGD`) interoperate with `compressed_tensors`
and upstream PyTorch/llm-compressor. Each section describes a specific
limitation, the code that bridges it locally, and the upstream change
that would remove the need for the workaround.

All workarounds live in the `learnable=True` path of `SpinQuantModifier`.
The `learnable=False` path (fixed-Hadamard SpinQuant introduced in v0.7.0)
is unchanged and uses upstream `apply_transform_config` directly.

---

## W1. Strict type equality in `apply_transform_weight`

**Library**: `compressed_tensors`

**Location in our code**: `src/llmcompressor/modifiers/transform/spinquant/factories.py:38-65`
(`_SubclassAwareHadamardTransform.forward`)

**Referenced upstream code**: `compressed_tensors/transform/utils/matrix.py:52-121`
(`apply_transform_weight` function body, particularly the `module_type ==
torch.nn.Linear` strict equality check at lines 102 and 112)

**Code snippet (our workaround)**:
```python
class _SubclassAwareHadamardTransform(HadamardTransform):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        ...
        module_type = self.module_type
        if issubclass(module_type, nn.Linear):
            module_type = nn.Linear
        elif issubclass(module_type, nn.Embedding):
            module_type = nn.Embedding
        return apply_transform_weight(
            weight.to(device=value.device),
            value.to(dtype=weight.dtype),
            self.args.location,
            module_type,
        ) / self._scale
```

**Justification**: when R1 is applied first via `parametrize.register_parametrization`,
the target module (e.g. `down_proj`) is replaced by a `ParametrizedLinear`
subclass of `nn.Linear`. A subsequent call to `apply_transform_weight` for
R2 then fails silently because the strict `module_type ==` check rejects
any subclass. Our widening logic replaces the stored `module_type` with the
base `nn.Linear` or `nn.Embedding` before the upstream call, which makes
the stacking of multiple transforms on the same module possible without
modifying upstream code.

**Suggested upstream fix**: replace `module_type == torch.nn.Linear` with
`issubclass(module_type, torch.nn.Linear)` (and the same for `nn.Embedding`)
inside `apply_transform_weight`, or expose a `module_type` override argument
so callers can pass the base class explicitly. Either change is two lines
and fully backward-compatible.

---

## W2. Shared weight cache forces R2 tying across layers

**Library**: `compressed_tensors`

**Location in our code**: `src/llmcompressor/modifiers/transform/spinquant/factories.py:78-115`
(`LearnableHadamardFactory.create_transform`, R2 branch at lines 86-99)

**Referenced upstream code**: `compressed_tensors/transform/factory/hadamard.py`
(`HadamardFactory.weights = ParameterizedDefaultDict(self._create_weight)`
cache keyed by `(size, dtype, device)`)

**Code snippet (our workaround)**:
```python
def create_transform(self, module: nn.Module, args: TransformArgs):
    ...
    # for R2: create a fresh weight per module so per-layer optimization
    # is possible. v_proj <-> o_proj tying is handled externally.
    if self.name == "R2":
        weight = self._create_weight(
            size, device=device, construct_device=exec_device, precision=precision,
        )
        return _SubclassAwareHadamardTransform(
            weight, None, self.scheme, args, type(module),
        )
```

**Justification**: the upstream `HadamardFactory` caches one matrix per
`(size, dtype, device)` triple and reuses it across every module that
references R2. This is correct and efficient for fixed Hadamard (all R2
instances are identical by construction), but it is incompatible with
per-layer Cayley training where each attention layer must converge to its
own distinct R2 matrix. Our override bypasses the cache specifically for
R2 by calling `_create_weight` directly, creating a fresh `nn.Parameter`
per module. R1/R3/R4 keep the upstream caching because they either have a
single global instance (R1) or are not learnable (R3/R4).

**Design choice rationale**: subclassing `HadamardFactory` to bypass the
per-size cache for R2 is the canonical pattern in `compressed_tensors` for
use cases requiring per-module weights. Adding a `tied: bool` field to
`TransformScheme` would expose internal cache semantics in the public API,
which is over-engineering for a constraint currently scoped to learnable
SpinQuant rotations.

---

## W3. `TransformScheme` has no within-layer tying option

**Library**: `compressed_tensors`

**Location in our code**: `src/llmcompressor/modifiers/transform/spinquant/base.py:435-474`
(method `_tie_r2_per_layer`)

**Referenced upstream code**: `compressed_tensors.transform.TransformScheme`
(no field for expressing that two schemes applied to paired modules must
share a single `Parameter`)

**Code snippet (our workaround)**:
```python
def _tie_r2_per_layer(self, model: nn.Module) -> None:
    ...
    for v_matches, o_matches in match_modules_set(
        model, (self.mappings.attn_v, self.mappings.attn_o)
    ):
        assert len(v_matches) == 1 and len(o_matches) == 1
        v_proj, o_proj = v_matches[0], o_matches[0]
        v_r2 = _find_r2(v_proj)
        if v_r2 is None:
            continue
        o_r2 = _find_r2(o_proj)
        if o_r2 is None or id(o_r2.weight) == id(v_r2.weight):
            continue
        del o_r2._parameters["weight"]
        o_r2.weight = v_r2.weight
```

**Justification**: SpinQuant math requires that R2 applied to `v_proj`
(output side) and its inverse applied to `o_proj` (input side, within the
same attention layer) share the same `Parameter`. Without this tying, the
two R2 matrices drift apart during Cayley training and the mathematical
invariance that keeps attention output unchanged breaks, producing a
non-functional model. W2 creates one fresh R2 per module, which is too
granular: we need one per layer, not one per module. This post-pass walks
the `v_proj`/`o_proj` pairs for each attention layer and replaces
`o_proj`'s R2 `Parameter` reference with `v_proj`'s R2 reference, so both
modules share one learnable matrix during Cayley backprop.

**Design choice rationale**: the post-pass tying in `_tie_r2_per_layer`
is the appropriate solution for a mathematical constraint specific to the
SpinQuant attention block invariance. Adding declarative paired-module
support to `TransformArgs` would require extending the matching API for a
constraint that does not generalize beyond this case.

---

## W4. `apply_transform_config` applies all schemes uniformly

**Library**: `compressed_tensors`

**Location in our code**:
- `src/llmcompressor/modifiers/transform/spinquant/base.py:216-239`
  (`on_start`, non-learnable filter + learnable branch split)
- `src/llmcompressor/modifiers/transform/spinquant/base.py:354-374`
  (`_apply_learnable_transform_config`, skip of non-learnable schemes)

**Referenced upstream code**: `compressed_tensors.transform.apply_transform_config`
(applies every scheme in the config through the same `TransformFactory`
code path; learnable and non-learnable are treated identically)

**Code snippet (our workaround)**:
```python
if self.learnable:
    non_learnable = {
        name: scheme
        for name, scheme in self.transform_config.config_groups.items()
        if not scheme.requires_grad
    }
    if non_learnable:
        apply_transform_config(
            model, TransformConfig(config_groups=non_learnable)
        )
        setattr(model, "transform_config", self.transform_config)
    self._apply_learnable_transform_config(model)
    self._tie_r2_per_layer(model)
    self._run_cayley_training(model)
```

**Justification**: the upstream `apply_transform_config` routes every
scheme through `HadamardFactory`, which in turn calls
`apply_transform_weight`. When R1 is `requires_grad=True` and has been
registered as a parametrization, any subsequent non-learnable scheme
(R3/R4) applied through the same path hits the strict type check (see W1)
on the now-`ParametrizedLinear` target. Our split first applies all
non-learnable schemes via the upstream path on the still-plain
`nn.Linear`, then hands off to `LearnableHadamardFactory` (with its W1
workaround) for the learnable ones. `_apply_learnable_transform_config`
skips non-learnable schemes via an explicit `if not scheme.requires_grad:
continue` because they have already been applied upstream.

**Suggested upstream fix**: once W1 is fixed in `compressed_tensors`, this
split becomes unnecessary: a single unified `apply_transform_config` call
would work for mixed learnable and non-learnable schemes. The presence of
this workaround is a direct consequence of W1 and resolves automatically
when W1 is resolved.

---

## W5. `oneshot()` enables accelerate offload, blocking learnable parametrize

**Library**: `llm-compressor` (upstream) + `compressed_tensors` interaction

**Location in our code**:
- `examples/transform/cayley_spinquant_train.py:59-64` (Phase 1 manual
  lifecycle bypass of `oneshot()`)
- `examples/transform/cayley_spinquant_quant.py:64-125` (Phase 2 load +
  bake via `load_rotated_model`, called before `oneshot()` in `main()`)

**Referenced upstream code**:
- `llmcompressor.entrypoints.oneshot` (wraps the model with accelerate
  `dispatch_model`, which in turn replaces parameter storage with
  `OffloadCache`)
- `compressed_tensors/transform/factory/base.py:145` explicit guard:
  `raise ValueError("Offloaded training is not supported")` when
  `scheme.requires_grad=True` and the module parameters are an
  `OffloadCache`

**Code snippet (our workaround)**:
```python
# Phase 1: manual lifecycle, no oneshot() wrapping
state = State(model=model)
modifier.on_initialize(state=state)
modifier.on_start(state=state, event=None)
modifier.on_end(state=state, event=None)
modifier.on_finalize(state=state)
torch.save(modifier.learned_rotations, "R_llmcompressor.bin")

# Phase 2: fresh model, load rotations, bake, then oneshot() for quant
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto").cuda()
# ... scaffold R1/R2, load R_llmcompressor.bin, bake
oneshot(
    model=model,
    recipe=[QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])],
    pipeline="datafree",
)
```

**Justification**: calling `oneshot()` on the learnable pipeline fails
immediately because `compressed_tensors` explicitly rejects
`register_parametrization` on `OffloadCache`-backed parameters. We
empirically verified this with an `oneshot` recipe combining
`SpinQuantModifier(learnable=True)` and `QuantizationModifier`, which
raises `ValueError: Offloaded training is not supported` at the first
call to `factory.apply_to_model`. `pipeline="datafree"` does not prevent
the offload wrapping. The two-phase workflow splits Cayley training
(lifecycle called manually, no offload) from quantization (`oneshot()` on
a fresh model where the trained rotations are baked into raw weights).

**Suggested upstream fix**: automatic detection of learnable schemes in the
recipe (skipping accelerate dispatch when any modifier requires gradient
flow on parametrizations) would be more aligned with `llm-compressor`'s
API conventions than an explicit boolean flag on `oneshot()`. The two-phase
manual workflow used here is the immediate workaround pending such
detection. An alternative path is to extend `compressed_tensors` to support
`register_parametrization` on offloaded parameters, but that is a larger
change than the dispatch-skip.

---

## W6. `HadamardFactory` stores unnormalized Hadamard weight

**Library**: `compressed_tensors`

**Location in our code**: `src/llmcompressor/modifiers/transform/spinquant/cayley_sgd.py:149-160, 200-202`
(`CayleySGD.step`, pre/post normalization + gradient rescaling)

**Referenced upstream code**:
- `compressed_tensors.transform.utils.hadamard.deterministic_hadamard_matrix`
  (returns entries in {+1, -1}, row norm = √n)
- `compressed_tensors.transform.factory.hadamard.HadamardTransform.forward`
  (divides by `self._scale = sqrt(n)` at each forward pass)

**Code snippet (our workaround)**:
```python
# Normalize X onto the Stiefel manifold (X @ X.T = I)
row_norms = X.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
X = X / row_norms

# Rescale gradient from stored-H space to normalized-X space:
# since X_eff = H / row_norms, ∂L/∂X_eff = ∂L/∂H * row_norms
G_t = (p.grad.data.to(torch.float64) * row_norms).T

# ... manifold update on X ...

# Restore row scale so HadamardTransform.forward (which divides by sqrt(n))
# still yields an orthogonal effective rotation matrix
p.data.copy_((X_new * row_norms).to(p.data.dtype))
```

**Justification**: Cayley SGD on the Stiefel manifold requires its
parameter to be orthonormal (`X @ X.T = I`). `HadamardFactory` however
stores the unnormalized Hadamard matrix H with entries in {+1, -1} and
row norm √n, and compensates by dividing the forward output by √n. If the
Cayley optimizer operates directly on H without normalization, the
effective step size is √n times too small (for R1 on Llama-3.1-8B,
n=4096 → factor 64), starving the rotation of any useful movement during
training. Our workaround normalizes the rows of the stored weight into
the Stiefel-valid form at the start of each step, performs the manifold
update on the normalized matrix, and restores the row scale when writing
back so `HadamardTransform.forward` continues to yield the correct
orthogonal rotation at inference time. The gradient is also rescaled by
`row_norms` to account for the chain rule `∂L/∂X_eff = ∂L/∂H × row_norms`.

**Design choice rationale**: the Stiefel manifold normalization is correctly
localized in `CayleySGD.step`. Per the chain rule, the gradient rescaling
by `row_norms` is mathematically tied to the optimizer's manifold
operations, not to the storage convention. The unnormalized convention
upstream serves the fixed Hadamard kernel path; changing it would risk
breaking existing checkpoints for a benefit that only the learnable path
would see.

---

## W7. `apply_transform_config` overwrites `model.transform_config`

**Library**: `compressed_tensors`

**Location in our code**: `src/llmcompressor/modifiers/transform/spinquant/base.py:226-234`
(setattr in `on_start` after partial `apply_transform_config` call)

**Referenced upstream code**: `compressed_tensors.transform.apply_transform_config`
(sets `model.transform_config = config` unconditionally with the sub-config
passed in, losing the full scope needed for downstream serialization)

**Code snippet (our workaround)**:
```python
if non_learnable:
    apply_transform_config(
        model, TransformConfig(config_groups=non_learnable)
    )
    # apply_transform_config overwrites model.transform_config with the
    # sub-config passed in. Restore the full config so config.json
    # serializes both learnable (R1/R2) and fixed (R3/R4) schemes for
    # downstream reconstruction.
    setattr(model, "transform_config", self.transform_config)
```

**Justification**: because of W4, we call `apply_transform_config` with
only the non-learnable subset of schemes. The upstream implementation
stores this sub-config on the model as the canonical
`model.transform_config`, which is then serialized into `config.json` at
`save_pretrained` time. If we leave the sub-config in place, the saved
model's metadata advertises only R3/R4 and forgets R1/R2, which breaks
any downstream reloader (vLLM, transformers, compressed_tensors HFQuantizer)
that would reconstruct the transforms from the serialized config. The
`setattr` restores the full `TransformConfig` covering all four rotations
after the partial apply has been consumed internally.

**Suggested upstream fix**: have `apply_transform_config` merge the passed
sub-config into an existing `model.transform_config` instead of
overwriting, or expose an `overwrite: bool = True` parameter so callers
that know they are applying a partial config can opt out of the overwrite.
The former is cleaner semantically (config is a monotonically growing
descriptor of applied transforms).

---

## W8. R4 must be applied before R1 parametrize registration

**Library**: `compressed_tensors`

**Location in our code**:
- `src/llmcompressor/modifiers/transform/spinquant/base.py:201-214`
  (condition in `on_start` triggering `_apply_r4` before the learnable
  transform_config)
- `src/llmcompressor/modifiers/transform/spinquant/base.py:376-412`
  (method `_apply_r4`)

**Referenced upstream code**:
- `compressed_tensors/transform/factory/base.py:120-128` (`_apply_to_module`,
  WEIGHT_INPUT/WEIGHT_OUTPUT branch): `update_offload_parameter(module,
  "weight", transform(module.weight))`
- `compressed_tensors/offload/__init__.py:140-144`
  (`update_offload_parameter` non-offloaded branch):
  `getattr(module, name).copy_(data)`

**Code snippet (our workaround)**:
```python
# In on_start, before the learnable transform_config block:
if self.learnable and "R4" not in self.rotations:
    self._apply_r4(model)

# _apply_r4 applies R4 via a dedicated apply_transform_config call with
# a hardcoded head_dim=128, while down_proj is still a plain nn.Linear:
def _apply_r4(self, model):
    r4_scheme = TransformScheme(
        type=self.transform_type, requires_grad=False, head_dim=128,
        apply=[
            TransformArgs(targets=[*self.mappings.mlp_out], location="input"),
            TransformArgs(targets=[*self.mappings.mlp_out],
                          location="weight_input", inverse=True),
        ],
    )
    apply_transform_config(model, TransformConfig(config_groups={"R4": r4_scheme}))
```

**Justification**: R4 modifies `down_proj.weight` by fusing R4^T into it
(location `weight_input`, inverse). The upstream fusion path calls
`update_offload_parameter(module, "weight", transform(module.weight))`,
which for a non-offloaded module resolves to
`getattr(module, "weight").copy_(data)`. If R1 has already been registered
as a parametrization on `down_proj.weight` (the learnable path attaches
R1 via `parametrize.register_parametrization`), then `getattr(module,
"weight")` returns the result of the R1 parametrize chain (a fresh tensor
computed as R1 @ W_original), and `.copy_(data)` modifies this transient
tensor. The underlying `module.parametrizations.weight.original` is
either left unchanged (copy lost) or re-routed through R1's
`right_inverse` depending on PyTorch's parametrize cache state, and the
end result observed empirically in early experiments was R4 effectively
applied twice in the forward chain, producing a silently incorrect model
with ~3× worse quantized PPL.

The safe order is to apply R4 first, while `down_proj.weight` is still a
plain `nn.Parameter`. Then the fusion path writes directly to the
parameter storage, and R1's subsequent parametrize registration sees the
already-R4-fused weight as its `original`. This ordering follows the
SpinQuant paper's canonical flow (Liu et al. 2024, arXiv:2405.16406):
norm fusion, then R3 and R4 fixed Hadamards, then R1 and R2 learnable
rotations trained via Cayley SGD against the quantized landscape. The
correct order is necessary to produce the reported PPL on
Llama-3.1-8B Base W4A16 G128. The `_apply_r4` helper hardcodes
`head_dim=128` so R4 uses the SpinQuant paper block size independently
of `self.transform_block_size` (which governs R1/R2/R3 sizing).

**Suggested upstream fix**: `update_offload_parameter` should detect
whether the target parameter has active parametrizations and either
(a) raise an explicit error directing the caller to apply the transform
before parametrize registration, or (b) route the update through the
parametrize chain coherently (e.g. via `module.parametrizations.weight.original`
direct write). Option (a) is simpler and makes the ordering constraint
explicit at the API surface. Option (b) would remove the constraint
entirely at the cost of a more complex code path inside
`update_offload_parameter`.

---

## Summary table

| # | Workaround | Library | File | Lines |
|---:|---|---|---|---:|
| W1 | Strict type equality | compressed_tensors | factories.py | 38-65 |
| W2 | Shared weight cache | compressed_tensors | factories.py | 78-115 |
| W3 | No tying option in TransformScheme | compressed_tensors | base.py | 435-474 |
| W4 | apply_transform_config uniform | compressed_tensors | base.py | 216-239, 354-374 |
| W5 | oneshot() blocks parametrize | llm-compressor | examples/ | (multiple) |
| W6 | Unnormalized Hadamard storage | compressed_tensors | cayley_sgd.py | 144-160, 200-202 |
| W7 | transform_config overwrite | compressed_tensors | base.py | 226-234 |
| W8 | R4 order before R1 parametrize | compressed_tensors | base.py | 201-214, 376-412 |

Of the 8 friction points, 5 (W1, W4, W5, W7, W8) suggest upstream changes
that would simplify the integration. For 3 (W2, W3, W6), the localized
workarounds in this PR are the canonical patterns and do not require
upstream changes. W5 lives in `llm-compressor` itself; the remaining
upstream-candidate items (W1, W4, W7, W8) live in `compressed_tensors`.
None require PyTorch changes.
