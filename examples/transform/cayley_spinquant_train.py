"""
Phase 1 of the SpinQuant Cayley workflow: train R1/R2 rotations.

This script learns the rotation matrices R1 and R2 on the Stiefel manifold
via Cayley SGD (Li et al. 2020, arXiv:2002.01113), following the SpinQuant
paper (Liu et al. 2024, arXiv:2405.16406).

Output: rotation state dict saved via ``torch.save()`` containing the
trained rotation matrices (1 R1 of size hidden_size x hidden_size, plus
one R2 of size head_dim x head_dim per attention layer).

Phase 2 (``cayley_spinquant_quant.py``) loads this file into a fresh model
and runs standard ``oneshot()`` quantization on the rotated weights.

Two-phase workflow: train rotations, then quantize with the learned
matrices. This split works around the limitation that ``oneshot()``
enables accelerate offload, which currently blocks
``register_parametrization`` for trainable transforms.

Usage:
    python3 cayley_spinquant_train.py
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import SpinQuantModifier

# Benchmarks in the PR description were measured on Meta-Llama-3.1-8B Base.
# This example uses Instruct to match the llm-compressor ecosystem convention.
# Switch to the Base variant to reproduce the exact PPL numbers from the description.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def main():
    logger.info("loading {}", MODEL_ID)
    # Load without device_map (default) and then .cuda() to put the full model
    # on GPU. This avoids accelerate's dispatch wrapping, which would block the
    # parametrize registration needed for gradient flow during Cayley training
    # (see INTEGRATION_WORKAROUNDS.md W5).
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    model = model.cuda()

    modifier = SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_type="hadamard",
        learnable=True,
        cayley_num_samples=800,
        cayley_batch_size=1,
        cayley_gradient_accumulation_steps=8,  # 100 optimizer steps (800 / 8)
        cayley_lr=1.5,
    )

    # manual lifecycle: equivalent of `oneshot()` but skips the accelerate
    # offload wrapping that blocks learnable transforms.
    state = State(model=model)
    modifier.on_initialize(state=state)
    modifier.on_start(state=state, event=None)
    modifier.on_end(state=state, event=None)
    modifier.on_finalize(state=state)

    # Save learned rotations to disk (modifier collects them in memory)
    output_path = Path("R_llmcompressor.bin")
    torch.save(modifier.learned_rotations, output_path)
    logger.info(
        "saved {} rotation matrices to {}",
        len(modifier.learned_rotations),
        output_path,
    )
    logger.info(
        "next step: run cayley_spinquant_quant.py to apply rotations to a fresh "
        "model, bake them into weights, and quantize."
    )


if __name__ == "__main__":
    main()
