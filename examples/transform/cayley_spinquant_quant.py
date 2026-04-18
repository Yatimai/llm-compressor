"""
Phase 2 of the SpinQuant Cayley workflow: apply trained R + quantize.

Loads the trained rotation matrices saved by ``cayley_spinquant_train.py``
into a fresh model, bakes them into the weights, then runs standard
``oneshot()`` quantization.

Because the rotations are baked in-place before ``oneshot()`` is called,
the quantization path sees plain ``nn.Linear`` modules and uses the standard
llm-compressor calibration / save pipeline -- no special support needed for
learnable transforms in oneshot.

Default: RTN with w_clip MSE grid search. See the PR description for
measured PPL on Llama-3.1-8B Base.

In our benchmarks on Llama-3.1-8B Base W4A16 G128, RTN with trained
rotations outperforms GPTQ. For GPTQ: replace QuantizationModifier
with GPTQModifier.

Environment variables:
    APPLY_R4: if "true", apply fixed Hadamard R4 on MLP for in-memory eval
              only (model cannot be saved with R4 active). Default: "false".

Usage:
    python3 cayley_spinquant_quant.py
    APPLY_R4=true python3 cayley_spinquant_quant.py   # ablation only
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

# Benchmarks in the PR description were measured on Meta-Llama-3.1-8B Base.
# This example uses Instruct to match the llm-compressor ecosystem convention.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
RBIN_PATH = "R_llmcompressor.bin"
OUTPUT_DIR = "Meta-Llama-3.1-8B-Instruct-Cayley-W4A16-G128"

APPLY_R4 = os.getenv("APPLY_R4", "false").lower() == "true"


def _strip_orphan_transforms(model: nn.Module) -> int:
    """Remove R1_*, R2_*, R4_* orphan submodules left by remove_parametrizations."""
    patterns = ("R1_", "R2_", "R4_")
    count = 0
    for module in model.modules():
        for attr_name in list(module._modules.keys()):
            if any(p in attr_name for p in patterns):
                delattr(module, attr_name)
                count += 1
    return count


def load_rotated_model() -> nn.Module:
    """Fresh model + Cayley rotations from R.bin baked into weights."""
    logger.info("loading fresh model")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto").cuda()

    logger.info("setting up rotation parametrizations (learnable scaffolding)")
    modifier = SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_type="hadamard",
        learnable=True,
        # placeholder: these Cayley hyperparameters are never read because we
        # skip _run_cayley_training below (direct call to helpers without
        # training loop)
        cayley_num_samples=8,
        cayley_batch_size=1,
        cayley_gradient_accumulation_steps=1,
    )
    state = State(model=model)
    modifier.on_initialize(state=state)
    # replicate on_start without _run_cayley_training: we just need the
    # parametrize scaffolding so the trained R values can be loaded into it.
    modifier.started_ = True
    from llmcompressor.utils import untie_word_embeddings

    untie_word_embeddings(model)
    modifier._center_embeddings(model)
    modifier._fuse_norms(model)
    modifier._apply_learnable_transform_config(model)
    modifier._tie_r2_per_layer(model)

    if not Path(RBIN_PATH).exists():
        raise FileNotFoundError(
            f"{RBIN_PATH} not found. Run cayley_spinquant_train.py first "
            f"to generate the rotation state dict."
        )
    logger.info("loading trained rotations from {}", RBIN_PATH)
    rbin = torch.load(RBIN_PATH, map_location="cpu")
    result = model.load_state_dict(rbin, strict=False)
    missing_rotations = set(rbin.keys()) & set(result.missing_keys)
    assert not missing_rotations, f"rotation keys missing: {missing_rotations}"
    # re-tie after load_state_dict (weights may have been overwritten with
    # independent copies)
    modifier._tie_r2_per_layer(model)

    logger.info("baking rotations into weights")
    for _, module in model.named_modules():
        if isinstance(
            module, (nn.Linear, nn.Embedding)
        ) and parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )

    if APPLY_R4:
        logger.info("APPLY_R4=true: applying fixed Hadamard R4 (in-memory only)")
        modifier._apply_r4(model)
    else:
        n = _strip_orphan_transforms(model)
        logger.info("stripped {} orphan R_* submodules from state_dict", n)

    return model


def main():
    model = load_rotated_model()

    logger.info("running RTN quantization (W4A16 G128, memoryless_mse observer)")
    oneshot(
        model=model,
        recipe=[
            QuantizationModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"],
                observer="memoryless_mse",
            ),
        ],
        output_dir=None if APPLY_R4 else OUTPUT_DIR,
    )

    if APPLY_R4:
        logger.info(
            "APPLY_R4=true: model evaluated in-memory, NOT saved "
            "(R4 hooks not persistable)"
        )
    else:
        logger.info("quantized model saved to {}", OUTPUT_DIR)


if __name__ == "__main__":
    main()
