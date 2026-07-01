from huggingface_hub import hf_hub_download
import os

repo_id = "HuggingFaceTB/SmolLM-135M-Instruct"
files = ["tokenizer.json", "config.json"]
target_dir = "models/nova_source"

for file in files:
    path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=target_dir)
    print(f"Downloaded {file} to {path}")
