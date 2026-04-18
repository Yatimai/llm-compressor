"""
WikiText-2 perplexity evaluation for the Cayley SpinQuant workflow.

Reproduces the evaluation protocol used for the benchmarks cited in the PR
description: WikiText-2 test split concatenated with double-newline
separators, split into non-overlapping 2048-token chunks (141 chunks for
the full split on a Llama-3 tokenizer), token-level perplexity computed
via ``exp(mean(nll))`` over the model's standard auto-regressive loss.

Usage:
    python eval_wikitext_ppl.py <model_path_or_id>

Example:
    python eval_wikitext_ppl.py Meta-Llama-3.1-8B-Instruct-Cayley-W4A16-G128
    python eval_wikitext_ppl.py meta-llama/Meta-Llama-3.1-8B
"""

import argparse

import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQUENCE_LENGTH = 2048


def eval_ppl_wikitext2(
    model, tokenizer, max_seq_len: int = MAX_SEQUENCE_LENGTH
) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts: list[str] = ds["text"]  # type: ignore[assignment]
    text = "\n\n".join(t for t in texts if t.strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    num_chunks = input_ids.shape[1] // max_seq_len
    logger.info("eval: {} chunks of {} tokens", num_chunks, max_seq_len)

    nlls = []
    for i in range(num_chunks):
        begin = i * max_seq_len
        end = begin + max_seq_len
        chunk = input_ids[:, begin:end].to(model.device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.float().item())

    return torch.exp(torch.tensor(nlls).mean()).item()


def main():
    parser = argparse.ArgumentParser(
        description="WikiText-2 PPL eval (141 chunks x 2048 tokens, token-level)."
    )
    parser.add_argument(
        "model_path",
        help="HuggingFace model id or local path to a (quantized) causal LM",
    )
    args = parser.parse_args()

    # Disable cuDNN SDPA on certain Blackwell/Hopper drivers where the fused
    # kernel produces slightly different logits than the reference path; keeps
    # PPL reproducible across hardware generations.
    torch.backends.cuda.enable_cudnn_sdp(False)

    logger.info("loading {}", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype="auto").cuda()
    model.eval()

    ppl = eval_ppl_wikitext2(model, tokenizer)
    logger.info("WikiText-2 PPL: {:.4f}", ppl)


if __name__ == "__main__":
    main()
