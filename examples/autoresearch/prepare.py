"""Prepare a tiny language-modeling task for autoresearch.

Downloads TinyStories from Hugging Face, tokenizes it with the GPT-2 tokenizer,
and saves compact train/val token tensors used by `train.py`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_SEQ_LEN = 256
TIME_BUDGET = 180
EVAL_BATCHES = 32
VOCAB_SIZE = 50257

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_NAME = "gpt2"
DATASET_NAME = "roneneldan/TinyStories"
TRAIN_TOKENS = 2_000_000
VAL_TOKENS = 200_000


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    @classmethod
    def from_directory(cls):
        return cls(AutoTokenizer.from_pretrained(TOKENIZER_NAME))

    def get_vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


def _tokenize_split(split: str, max_tokens: int, tokenizer: Tokenizer) -> torch.Tensor:
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    tokens: list[int] = []
    for row in ds:
        text = row.get("text") or row.get("story") or ""
        if not text:
            continue
        tokens.extend(tokenizer.encode(text))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= max_tokens:
            break
    if len(tokens) < MAX_SEQ_LEN + 1:
        raise RuntimeError(f"not enough tokens for split={split}")
    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


def prepare() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.pt"
    val_path = DATA_DIR / "val.pt"
    if train_path.exists() and val_path.exists():
        print(f"Data already prepared in {DATA_DIR}")
        return

    tokenizer = Tokenizer.from_directory()
    print(f"Tokenizing {DATASET_NAME} with {TOKENIZER_NAME}")
    train = _tokenize_split("train", TRAIN_TOKENS, tokenizer)
    val = _tokenize_split("validation", VAL_TOKENS, tokenizer)
    torch.save(train, train_path)
    torch.save(val, val_path)
    print(f"Saved train tokens: {train.numel():,} -> {train_path}")
    print(f"Saved val tokens:   {val.numel():,} -> {val_path}")


def _load_tokens(split: str, device: str | None = None) -> torch.Tensor:
    path = DATA_DIR / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare.py first")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=device)


def make_dataloader(batch_size: int, seq_len: int, split: str, seed: int | None = None):
    assert split in {"train", "val"}
    data = _load_tokens(split)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=data.device).manual_seed(seed)
    while True:
        ix = torch.randint(0, data.numel() - seq_len - 1, (batch_size,), device=data.device, generator=generator)
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        yield x, y


EVAL_SEED = 1234


@torch.no_grad()
def evaluate_bpb(model, batch_size: int, seq_len: int) -> float:
    model.eval()
    # Fixed seed so every trial is scored on the *same* val batches -> the
    # leaderboard measures real improvements, not val-sampling noise.
    loader = make_dataloader(batch_size, seq_len, "val", seed=EVAL_SEED)
    tokenizer = Tokenizer.from_directory()
    token_bytes = torch.tensor(
        [
            0 if token_id == tokenizer.eos_token_id else len(tokenizer.decode([token_id]).encode("utf-8"))
            for token_id in range(tokenizer.get_vocab_size())
        ],
        device=next(model.parameters()).device,
    )
    total_nats = 0.0
    total_bytes = 0
    for _ in range(EVAL_BATCHES):
        x, y = next(loader)
        loss = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        bytes_flat = token_bytes[y_flat]
        valid = bytes_flat > 0
        total_nats += float(loss[valid].sum().item())
        total_bytes += int(bytes_flat[valid].sum().item())
    model.train()
    return total_nats / max(1, total_bytes) / math.log(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare autoresearch data")
    parser.parse_args()
    prepare()
