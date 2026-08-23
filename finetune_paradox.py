"""Conservative 4-bit QLoRA finetuning for the Adam assistant.

This script trains only the assistant response tokens, evaluates on a held-out
split, and restores the best checkpoint to reduce memorization.
"""

import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("DATA_PATH", "adam_alpaca.parquet")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./adam-tinyllama-qlora")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
EPOCHS = float(os.getenv("EPOCHS", "3"))


def conversation(example):
    user_content = str(example["instruction"]).strip()
    input_text = str(example.get("input", "") or "").strip()
    if input_text:
        user_content += f"\n\nContext:\n{input_text}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(example["output"]).strip()},
    ]


def tokenize_example(example, tokenizer):
    messages = conversation(example)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The selected tokenizer has no chat template.")

    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    labels = list(full["input_ids"])
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    full["labels"] = labels
    return full


class EvaluationMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "eval_loss" in logs:
            print(
                f"[EVAL] step={state.global_step} "
                f"validation_loss={float(logs['eval_loss']):.4f}"
            )


def main():
    use_cuda = torch.cuda.is_available()
    print(f"Device mode: {'CUDA 4-bit QLoRA' if use_cuda else 'CPU LoRA fallback'}")
    if not use_cuda:
        print("WARNING: CPU fallback is much slower and needs substantial RAM; use a GPU for the full run.")

    print(f"Loading reviewed Parquet data from {DATA_PATH}...")
    dataframe = pd.read_parquet(DATA_PATH)
    required = {"instruction", "input", "output"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"Train examples: {len(split['train'])}; validation examples: {len(split['test'])}")

    print(f"Loading base model {MODEL_ID}...")
    compute_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
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

    model_kwargs = {"torch_dtype": compute_dtype if use_cuda else torch.float32}
    if use_cuda:
        model_kwargs.update(quantization_config=quantization_config, device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.config.use_cache = False
    if use_cuda:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    tokenized = split.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Formatting response-only chat examples",
    )

    use_bf16 = compute_dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=EPOCHS,
        max_steps=-1,
        warmup_steps=2,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        gradient_checkpointing=use_cuda,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        ),
        callbacks=[EvaluationMonitor(), EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting conservative 1.1B response-only training...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final validation loss: {metrics.get('eval_loss')}")
    print(f"Saving best Adam adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Finished. The output is a LoRA adapter plus tokenizer files.")


def tokenize_batch(batch, tokenizer):
    rows = []
    keys = list(batch.keys())
    for index in range(len(batch[keys[0]])):
        rows.append({key: batch[key][index] for key in keys})
    encoded = [tokenize_example(row, tokenizer) for row in rows]
    return {
        "input_ids": [item["input_ids"] for item in encoded],
        "attention_mask": [item["attention_mask"] for item in encoded],
        "labels": [item["labels"] for item in encoded],
    }


if __name__ == "__main__":
    main()
