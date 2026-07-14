import os
import torch
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# ── HuggingFace Auth ────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("WARNING: No HF_TOKEN set. Set it with: export HF_TOKEN=hf_xxxxx")

# ── Configuration ────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM3-3B-Base"
DATA_PATH = "adam_alpaca.parquet"
OUTPUT_DIR = "./adam-finetuned"

def format_alpaca(example):
    if example["input"]:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n"
    
    full_text = prompt + example["output"] + "<|endoftext|>"
    return {"text": full_text}

def main():
    # 1. Load Parquet Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    dataset = Dataset.from_pandas(df)
    
    # 2. Load Model & Tokenizer
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    
    # 3. Preprocess Dataset
    def tokenize_function(examples):
        res = format_alpaca(examples)
        tokenized = tokenizer(
            res["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # 4. Training Arguments
    # Note: User requested 1300-2000 monitor to avoid overfitting.
    # We'll set max_steps to a bit more than 2000 so we can observe that range.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=10,
        max_steps=2100,
        save_steps=500,
        eval_strategy="no",
        fp16=True,
        bf16=False,
        push_to_hub=False,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        gradient_checkpointing=True,
    )

    # 5. Custom Trainer for Monitoring
    class MonitorTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Custom monitoring for steps 1300-2000
            if 1300 <= self.state.global_step <= 2000:
                if self.state.global_step % 100 == 0:
                    print(f"\n[MONITOR] Step {self.state.global_step}: Monitoring loss to prevent overfitting. Current Loss: {loss.item():.4f}")
            
            # Stop if we reach the target verification point

                
            return loss

    # 6. Initialize Trainer
    trainer = MonitorTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    # 7. Run Training
    print("Starting training...")
    try:
        trainer.train()
    finally:
        print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
