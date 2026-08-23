import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# TinyLlama is approximately 1.1B parameters, is openly downloadable, and is
# small enough for practical QLoRA finetuning on a Colab T4/L4 GPU.
MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("DATA_PATH", "adam_alpaca.parquet")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./adam-tinyllama-qlora")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2100"))


def format_example(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "") or "").strip()
    output = str(example.get("output", "")).strip()
    context = f"\n\n### Input:\n{input_text}" if input_text else ""
    prompt = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}{context}\n\n"
        f"### Response:\n{output}"
    )
    return prompt + "<|eos|>"


class RangeMonitorCallback(TrainerCallback):
    """Print loss checkpoints in the requested 1300–2000 step range."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not (1300 <= state.global_step <= 2000):
            return
        if state.global_step % 100 == 0:
            loss = logs.get("loss")
            if loss is not None:
                print(
                    f"[MONITOR] Adam QLoRA step {state.global_step}: "
                    f"loss={float(loss):.4f}; inspect for overfitting."
                )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit bitsandbytes QLoRA requires a CUDA GPU for this setup. "
            "In Colab, select Runtime > Change runtime type > T4 GPU."
        )

    print(f"Loading Parquet data from {DATA_PATH}...")
    dataframe = pd.read_parquet(DATA_PATH)
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)

    print(f"Loading 4-bit base model {MODEL_ID}...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_batch(batch):
        texts = [format_example(item) for item in dataset_from_batch(batch)]
        return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

    # Convert the column-wise batch supplied by datasets.map into row mappings.
    def dataset_from_batch(batch):
        keys = list(batch.keys())
        return [
            {key: batch[key][index] for key in keys}
            for index in range(len(batch[keys[0]]))
        ]

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing Adam dataset",
    )

    use_bf16 = compute_dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[RangeMonitorCallback()],
    )

    print("Starting 4-bit QLoRA finetuning for Adam...")
    trainer.train()
    print(f"Saving LoRA adapter and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Finished successfully. This directory contains the adapter, not a merged full model.")


if __name__ == "__main__":
    main()
