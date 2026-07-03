import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext_subset(num_samples=1000, seq_len=128):
    """
    Downloads a subset of WikiText-2 and tokenizes it.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty lines
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    # Take subset
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len + 1, padding="max_length")
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    return tokenized, tokenizer

class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data["input_ids"]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        x = self.data[i][:-1]
        y = self.data[i][1:]
        return x, y
