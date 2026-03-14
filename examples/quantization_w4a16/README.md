# `int4` Weight Quantization

`llm-compressor` supports quantizing weights to `int4` for memory savings and inference acceleration with `vLLM`

> `int4` mixed precision computation is supported on Nvidia GPUs with compute capability > 8.0 (Ampere, Ada Lovelace, Hopper).

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-W4A16-G128` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are four steps:
1) Load model
2) Prepare calibration data
3) Apply quantization
4) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM` for handling quantized saving and loading. 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Prepare Calibration Data

Prepare the calibration data. When quantizing weigths of a model to `int4` using GPTQ, we need some sample data to run the GPTQ algorithms. As a result, it is very useful to use calibration data that closely matches the type of data used in deployment. If you have fine-tuned a model, using a sample of your training data is a good idea.

In our case, we are quantizing an Instruction tuned generic model, so we will use the `ultrachat` dataset. Some best practices include:
* 512 samples is a good place to start (increase if accuracy drops)
* 2048 sequence length is a good place to start
* Use the chat template or instrucion template that the model is trained with

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 3) Apply Quantization

With the dataset ready, we will now apply quantization.

We first select the quantization algorithm.

In our case, we will apply the default GPTQ recipe for `int4` (which uses static group size 128 scales) to all linear layers.
> See the `Recipes` documentation for more information on making complex recipes

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Configure the quantization algorithm to run.
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model, dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `int4` model!

### 4) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
```

We can evaluate accuracy with `lm_eval` (`pip install lm_eval==v0.4.3`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

We can see the resulting scores look good!

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.728|±  |0.0282|
|     |       |strict-match    |     5|exact_match|↑  |0.720|±  |0.0285|
```

---

## iMatrix Importance-Weighted Quantization

`imatrix_mse` is an observer that uses per-channel activation importance (E[x²]) to weight quantization error during range selection. Channels that carry more signal get more careful range optimization.

Two components work together:
- **`IMatrixGatherer`**: collects E[x²] per input channel via forward pre-hooks during calibration
- **`imatrix_mse` observer**: extends the MSE grid search with importance weighting: `err = sum(importance * |Q(w) - w|^p)`

> See [RFC #2456](https://github.com/vllm-project/llm-compressor/discussions/2456) for the full design discussion.

### Usage

```bash
python3 llama3_imatrix_example.py
```

The simplest setup uses `preset_name_to_scheme` to configure W4A16 and swaps in the `imatrix_mse` observer:

```python
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    IMatrixGatherer(ignore=["lm_head"]),
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

### Composing with AWQ and GPTQ

iMatrix is additive — it composes with any quantization method:

```python
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.gptq import GPTQModifier

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"]),
    IMatrixGatherer(ignore=["lm_head"]),
    GPTQModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

### Results (Llama-3.1-8B, WikiText PPL)

**W4A16, group_size=128:**

| Config | PPL |
|---|---|
| RTN `memoryless_minmax` | 6.78 |
| RTN `imatrix_mse` | 6.76 |
| GPTQ | 6.58 |
| AWQ + GPTQ + `imatrix_mse` | 6.52 |

**W4A16, group_size=32:**

| Config | PPL |
|---|---|
| RTN `memoryless_minmax` gs128 | 6.78 |
| RTN `imatrix_mse` gs32 | 6.46 |
| AWQ + GPTQ + `imatrix_mse` gs32 | 6.39 |

RTN `imatrix_mse` gs32 (6.46) outperforms GPTQ gs128 (6.58) in ~40s vs ~800s.

### Observer Parameters

The observer accepts optional `observer_kwargs` for fine-tuning:

| Parameter | Default | Description |
|---|---|---|
| `norm` | 2.4 | Error exponent (`\|Q(w) - w\|^norm`) |
| `maxshrink` | 0.20 | Max fraction to shrink the range |
| `grid` | 20 | Number of grid search steps |
| `patience` | 5 | Early stopping after N steps without improvement |
| `maxgrow` | 0.0 | Max fraction to grow the range beyond observed min/max |

The defaults work well for most cases. To customize:

```python
scheme.weights.observer_kwargs = {"maxgrow": 0.10}
```

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
