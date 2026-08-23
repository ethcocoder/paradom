# Google Colab: Adam 1.1B Full-Precision LoRA Finetuning

This guide trains an **Adam** adapter on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The base model is approximately 1.1B parameters and is trained with **full-precision LoRA**. The output is a small LoRA adapter, while the base model remains unchanged.

## 1. Create a GPU notebook

Open [Google Colab](https://colab.research.google.com/), create a new notebook, then choose **Runtime → Change runtime type → T4 GPU**. Confirm the GPU before training:

```python
!nvidia-smi
```

The script automatically uses CUDA when available and falls back to CPU. A GPU is strongly recommended because full-precision CPU training is slow and memory-intensive.

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

The expected columns are `instruction`, `input`, `output`, `category`, and `quality`. The reviewed builder creates 47 unique examples across 10 categories. The training script creates a deterministic 80/20 validation split; expand the dataset further for production use.

## 5. Run a short smoke test first

Run one epoch to confirm that the device selection, PEFT, tokenizer, chat template, Parquet loader, validation, and response-only labels work:

```python
%cd /content/paradom
!EPOCHS=1 OUTPUT_DIR=./adam-smoke-test python finetune_paradox.py
```

You should see a trainable-parameter report and a successful save under `adam-smoke-test`. The base model is loaded in full precision, while only LoRA parameters are updated.

## 6. Run the actual training

For the configured run, use a maximum of three epochs. The script evaluates every five steps, keeps the best validation checkpoint, and stops after three evaluations without improvement:

```python
%cd /content/paradom
!EPOCHS=3 OUTPUT_DIR=./adam-tinyllama-qlora python finetune_paradox.py
```

The script uses `warmup_steps=2`, a held-out validation split, response-only labels, and early stopping to reduce memorization. The output directory contains the LoRA adapter and tokenizer, not the full base model. Keep the runtime connected until the final save completes.

To stop after confirming that training is operating correctly, interrupt the cell. Checkpoints are saved every five evaluation steps, and only the two newest are retained. Resume from the most recent checkpoint when needed with:

```python
%cd /content/paradom
!EPOCHS=3 OUTPUT_DIR=./adam-tinyllama-qlora python finetune_paradox.py
```

## 7. Test Adam in Colab

Create and run this test cell. It loads the base model in full precision and applies the trained LoRA adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "/content/paradom/adam-tinyllama-qlora"

use_cuda = torch.cuda.is_available()
dtype = (
    torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
    else torch.float16 if use_cuda
    else torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model_kwargs = {"torch_dtype": dtype}
if use_cuda:
    model_kwargs["device_map"] = "auto"
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

questions = ["What is your name?", "Who are you?", "What is your purpose?"]
for question in questions:
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
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

The expected identity answer should mention **Adam**. Also test unseen prompts, because low training loss alone does not prove generalization.

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

If the script reports that CUDA is unavailable, it will use the CPU fallback; selecting a GPU runtime is recommended. If the model is out of memory, reduce `MAX_LENGTH` to 256 and change `per_device_train_batch_size` from 2 to 1 in `finetune_paradox.py`.

If validation loss stops improving or responses become repetitive, use the best saved checkpoint, reduce epochs or learning rate, and add more varied reviewed examples. Validation loss and early stopping are the primary overfitting controls.

## Files changed in `v3`

| File | Purpose |
|---|---|
| `finetune_paradox.py` | TinyLlama 1.1B, full-precision LoRA, response-only labels, validation, and early stopping |
| `create_dataset.py` | Generates the Adam Alpaca-format Parquet dataset |
| `docs/colab-instruction.md` | This complete Colab procedure |
| `.github/workflows/finetune.yml` | Automated workflow dependency updates and artifact upload |
      
