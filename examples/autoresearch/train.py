"""Small editable GPT training script for autoresearch.

The agent is allowed to modify this file. `prepare.py` owns the fixed dataset,
batching, and validation metric.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = MAX_SEQ_LEN
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        bsz, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.seq_len, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, reduction: str = "mean"):
        bsz, seq_len = idx.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](pos)[None, :, :]
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction,
        )
        return loss.view_as(targets) if reduction == "none" else loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Hyperparameters: agents should edit these first.
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
MIN_LR_FRAC = 0.1
COMPILE_MODEL = True


def lr_multiplier(step: int, progress: float) -> float:
    """Warmup by step, then cosine decay over `progress` in [0, 1].

    `progress` is the fraction of the wall-clock training budget elapsed, so the
    schedule reaches MIN_LR_FRAC exactly at TIME_BUDGET no matter how many steps
    actually run. Do not tie the horizon to a hardcoded step estimate.
    """
    if step < WARMUP_STEPS:
        return max(1, step) / WARMUP_STEPS
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cosine


def main() -> None:
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    tokenizer = Tokenizer.from_directory()
    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    print("--- setup ---", flush=True)
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"cuda: {torch.cuda.get_device_name()}", flush=True)
    print(f"config: {asdict(config)}", flush=True)
    print(f"batch_size: {BATCH_SIZE}", flush=True)

    model = GPT(config).to(device)
    num_params = model.num_params()
    if COMPILE_MODEL:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    train_loader = make_dataloader(BATCH_SIZE, MAX_SEQ_LEN, "train")

    t_start = time.time()
    train_seconds = 0.0
    step = 0
    smooth_loss = None
    print("--- training ---", flush=True)
    while train_seconds < TIME_BUDGET:
        t0 = time.time()
        x, y = next(train_loader)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        mult = lr_multiplier(step, (time.time() - t_start) / TIME_BUDGET)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * mult
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_seconds = time.time() - t_start
        loss_f = float(loss.item())
        smooth_loss = loss_f if smooth_loss is None else 0.9 * smooth_loss + 0.1 * loss_f
        if step % 10 == 0:
            print(
                f"step {step:05d} | loss {loss_f:.4f} | smooth {smooth_loss:.4f} | "
                f"lr {LEARNING_RATE * mult:.2e} | dt {dt*1000:.0f}ms | elapsed {train_seconds:.0f}s",
                flush=True,
            )
        if not math.isfinite(loss_f):
            raise RuntimeError(f"non-finite loss: {loss_f}")
        step += 1

    print("--- eval ---", flush=True)
    with autocast:
        val_bpb = evaluate_bpb(model, BATCH_SIZE, MAX_SEQ_LEN)
    total_tokens = step * BATCH_SIZE * MAX_SEQ_LEN
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {config.n_layer}")


if __name__ == "__main__":
    main()
