"""
Integration test: IMatrixGatherer -> QuantizationModifier(imatrix_mse).

Uses nm-testing/tinysmokellama-3.2, the standard tiny HF model in this repo.
No GPU required — runs on CPU.

The oneshot() entrypoint requires a PreTrainedModel (not a plain nn.Module)
and a HuggingFace Dataset or dataset string (not a list of dicts), so we
use the real entrypoint with minimal calibration samples.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

MODEL_ID = "nm-testing/tinysmokellama-3.2"
DATASET = "open_platypus"
NUM_CALIB_SAMPLES = 4
MAX_SEQ_LEN = 128


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the tiny model and tokenizer once per module."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def _get_linear_layer_names(model, ignore=None):
    """Return names of all nn.Linear modules not in the ignore list."""
    ignore = ignore or []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not any(pattern in name for pattern in ignore):
                names.append(name)
    return names


def _get_module_by_name(model, name):
    """Retrieve a submodule by dotted name."""
    parts = name.split(".")
    m = model
    for part in parts:
        m = getattr(m, part)
    return m


class TestGathererObserverIntegration:
    """IMatrixGatherer + imatrix_mse observer end-to-end."""

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_pipeline_produces_quantized_model(self):
        """Gatherer collects importance, observer consumes it, model is quantized."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                            "observer_kwargs": {
                                "norm": 2.4,
                                "maxshrink": 0.7,
                                "maxgrow": 0.10,
                            },
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=DATASET,
            splits={"calibration": "train[:5%]"},
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        # Verify importance was cleaned up after finalization
        targeted_names = _get_linear_layer_names(model, ignore=["lm_head"])
        for name in targeted_names:
            mod = _get_module_by_name(model, name)
            assert not hasattr(
                mod, "_imatrix_importance"
            ), f"{name} should not have _imatrix_importance after finalization"

        # Hooks should be cleaned up
        total_hooks = sum(len(m._forward_pre_hooks) for m in model.modules())
        assert (
            total_hooks == 0
        ), f"Expected 0 forward pre-hooks after completion, found {total_hooks}"

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_gatherer_without_observer_no_crash(self):
        """IMatrixGatherer alone should run without error and clean up after."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [IMatrixGatherer(ignore=["lm_head"])]

        oneshot(
            model=model,
            dataset=DATASET,
            splits={"calibration": "train[:5%]"},
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        # Importance should be cleaned up after finalization
        targeted_names = _get_linear_layer_names(model, ignore=["lm_head"])
        assert len(targeted_names) > 0, "Model should have targeted Linear layers"

        for name in targeted_names:
            mod = _get_module_by_name(model, name)
            assert not hasattr(
                mod, "_imatrix_importance"
            ), f"{name} should not have _imatrix_importance after finalization"

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_pipeline_with_regex_targets(self):
        """Gatherer + observer with regex targets for specific attention projections."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            IMatrixGatherer(
                ignore=["lm_head"],
                targets=["re:.*self_attn.(q|k|v)_proj$"],
            ),
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["re:.*self_attn.(q|k|v)_proj$"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=DATASET,
            splits={"calibration": "train[:5%]"},
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        # Verify importance was cleaned up after finalization
        for name, module in model.named_modules():
            assert not hasattr(
                module, "_imatrix_importance"
            ), f"{name} should not have _imatrix_importance after finalization"

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_observer_without_gatherer_fallback(self):
        """
        imatrix_mse observer without a gatherer should fall back to uniform MSE
        (no _imatrix_importance available), completing without error.
        """
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        # Should complete without error, falling back to uniform weighting
        oneshot(
            model=model,
            dataset=DATASET,
            splits={"calibration": "train[:5%]"},
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )
