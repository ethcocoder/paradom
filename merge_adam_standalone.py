"""Merge the Adam LoRA adapter into the TinyLlama 1.1B base model."""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    print(f"Merging on CPU with {args.cpu_threads} PyTorch threads")
    print(f"Base model: {args.base_model}")
    print(f"Adam adapter: {args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    adapter_model = PeftModel.from_pretrained(base, args.adapter)
    merged_model = adapter_model.merge_and_unload()
    merged_model.config.use_cache = True

    os.makedirs(args.output, exist_ok=True)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(args.output)
    print(f"Saved standalone model to {args.output}")


if __name__ == "__main__":
    main()

