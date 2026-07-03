import os
import sys
# Debug PYTHONPATH for Colab troubleshooting
print(f"DEBUG: sys.path is {sys.path}")
print(f"DEBUG: CWD is {os.getcwd()}")
try:
    import paradom
    print(f"DEBUG: paradom found at {paradom.__file__}")
except ImportError as e:
    print(f"DEBUG: paradom NOT found: {e}")

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from paradom.models.tiny_transformer import TinyTransformer
from paradom.models.tiny_mamba import TinyMamba
from scripts.data_utils import get_wikitext_subset, CausalDataset

SEED = 42
NUM_SAMPLES = 10000
EPOCHS = 5
BATCH_SIZE = 16
LR = 5e-4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, optimizer, device, epochs=EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    tokenized, _ = get_wikitext_subset(num_samples=NUM_SAMPLES)
    dataset = CausalDataset(tokenized)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\nTraining on {NUM_SAMPLES} WikiText-2 samples, {EPOCHS} epochs")

    print("\n--- Training Transformer (Source) ---")
    transformer = TinyTransformer().to(device)
    optimizer_t = torch.optim.AdamW(transformer.parameters(), lr=LR)
    train(transformer, loader, optimizer_t, device)
    torch.save(transformer.state_dict(), "checkpoints/tiny_transformer_trained.pt")

    print("\n--- Training Mamba (Target Baseline) ---")
    mamba = TinyMamba().to(device)
    optimizer_m = torch.optim.AdamW(mamba.parameters(), lr=LR)
    train(mamba, loader, optimizer_m, device)
    torch.save(mamba.state_dict(), "checkpoints/tiny_mamba_trained.pt")

    print("\nTraining complete. Checkpoints saved to checkpoints/")


if __name__ == "__main__":
    main()
