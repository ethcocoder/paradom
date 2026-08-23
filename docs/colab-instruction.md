# Google Colab: Adam 1.1B Full-Precision LoRA Finetuning

This guide trains an **Adam** adapter on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The base model is approximately 1.1B parameters and is trained with **full-precision LoRA**. The output is a small LoRA adapter, while the base model remains unchanged.

## 1. Create a CPU notebook

Open [Google Colab](https://colab.research.google.com/) and create a new notebook. Keep the runtime on **CPU**; this project intentionally does not check for or use CUDA. The full-precision 1.1B model is memory-intensive, so use a runtime with as much RAM and CPU capacity as available.

## 2. Clone the `v3` branch

Run this cell:

```python
%cd /content
!rm -rf paradom
!git clone --branch v3 --single-branch https://github.com/ethcocoder/paradom.git
%cd /content/paradom
!git status
```

## 3. Install the training dependencies

```python
!pip install -U "transformers>=4.45" "datasets>=2.20" "accelerate>=1.1" peft pandas pyarrow sentencepiece safetensors
```

After installation, restart the runtime only if Colab requests it. If you restart, rerun the clone cell and return to `/content/paradom`.

## 4. Generate and inspect the Parquet dataset

```python
%cd /content/paradom
!python create_dataset.py
```

Verify the columns and sample count:

```python
import pandas as pd

frame = pd.read_parquet("adam_alpaca.parquet")
print(frame.shape)
print(frame.columns.tolist())
print(frame.head(3).to_string(index=False))
```

The expected columns are `instruction`, `input`, `output`, `category`, and `quality`. The builder creates exactly 2,000 unique examples across 13 categories, including identity, creator, GitHub, curiosity, philosophy, reasoning, coding, safety, and evaluation. The examples teach that Adam Natnael was built by Natnael Ermiyas, who is described as 20 years old at the time of the project, and reference the project repository at `https://github.com/ethcocoder/paradom`. The training script creates a deterministic 80/20 validation split.

## 5. Run a short smoke test first

Run one epoch to confirm CPU execution, PEFT, tokenizer, chat template, Parquet loading, validation, and response-only labels:

```python
%cd /content/paradom
!CPU_THREADS=$(nproc) EPOCHS=1 OUTPUT_DIR=./adam-smoke-test python finetune_paradox.py
```

You should see a trainable-parameter report and a successful save under `adam-smoke-test`. The base model is loaded in full precision, while only LoRA parameters are updated.

## 6. Run the actual training

For the configured run, use a maximum of three epochs. The script evaluates every five steps, keeps the best validation checkpoint, and stops after three evaluations without improvement:

```python
%cd /content/paradom
!CPU_THREADS=$(nproc) EPOCHS=3 OUTPUT_DIR=./adam-tinyllama-qlora python finetune_paradox.py
```

The script uses `warmup_steps=2`, a held-out validation split, response-only labels, and early stopping to reduce memorization. The output directory contains the LoRA adapter and tokenizer, not the full base model. Keep the runtime connected until the final save completes.

To stop after confirming that training is operating correctly, interrupt the cell. Checkpoints are saved every five evaluation steps, and only the two newest are retained. Resume from the most recent checkpoint when needed with:

```python
%cd /content/paradom
!CPU_THREADS=$(nproc) EPOCHS=3 OUTPUT_DIR=./adam-tinyllama-qlora python finetune_paradox.py
```

## 7. Test Adam in Colab

Create and run this test cell. It loads the base model in full precision and applies the trained LoRA adapter:

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "/content/paradom/adam-tinyllama-qlora"

CPU_THREADS = os.cpu_count() or 1
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map={"": "cpu"},
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

questions = [
    "What is your name?",
    "Who built you and what is your GitHub project?",
    "What question are you asking yourself about curiosity?",
]
for question in questions:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Q: {question}\nA: {answer.strip()}\n")
```

The expected identity answer should mention **Adam Natnael**, attribute construction to **Natnael Ermiyas**, and provide the GitHub repository accurately when asked. Curiosity tests should produce a thoughtful follow-up question without claiming human consciousness. Also test unseen prompts, because low training loss alone does not prove generalization.

## 8. Download the adapter

Download the adapter directory as a zip file:

```python
!cd /content && zip -r adam-tinyllama-qlora.zip paradom/adam-tinyllama-qlora
from google.colab import files
files.download("/content/adam-tinyllama-qlora.zip")
```

To run it later, keep the base model ID unchanged and place the downloaded adapter directory at the path used by the inference script. The base model is downloaded separately from Hugging Face the first time it is loaded.

## 9. Optional: merge the adapter

Merging creates a larger standalone model and requires substantially more memory. For an 8GB machine, keep the adapter separate. Full-precision loading may require more than 8GB once training or generation overhead is included.

## 10. Common errors

The workflow and scripts are CPU-only and do not verify CUDA. They use all available CPU threads by default. If the model is out of memory, reduce `MAX_LENGTH` to 256 and change `per_device_train_batch_size` from 2 to 1 in `finetune_paradox.py`. You can cap threads with `CPU_THREADS=4`.

If validation loss stops improving or responses become repetitive, use the best saved checkpoint, reduce epochs or learning rate, and add more varied reviewed examples. Validation loss and early stopping are the primary overfitting controls.

## Files changed in `v3`

| File | Purpose |
|---|---|
| `finetune_paradox.py` | TinyLlama 1.1B, full-precision LoRA, response-only labels, validation, and early stopping |
| `create_dataset.py` | Generates the 2,000-example Adam persona Parquet dataset |
| `docs/colab-instruction.md` | This complete Colab procedure |
| `.github/workflows/finetune.yml` | Automated workflow dependency updates and artifact upload |
      
