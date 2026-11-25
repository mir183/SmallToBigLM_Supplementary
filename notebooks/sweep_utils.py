"""Shared data + training utilities for the lever-comparison sweep.

Importing this module loads TinyShakespeare (char-level) once and exposes
train_config(...) which trains one architecture variant and returns metrics.
Fairness controls:
  - identical train-batch order across configs for a given seed (train_gen)
  - identical eval batches for ALL configs (fixed eval generator seed)
"""
import os
import math
import time
from contextlib import nullcontext

import torch
from model import Config, GPT, count_params

device = "cuda" if torch.cuda.is_available() else "cpu"

_cands = [
    "../dataset/tinyshakespeare/input.txt",
    "dataset/tinyshakespeare/input.txt",
]
_path = next(p for p in _cands if os.path.exists(p))
_text = open(_path, encoding="utf-8").read()
chars = sorted(set(_text))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
_data = torch.tensor([stoi[c] for c in _text], dtype=torch.long)
_n = int(0.9 * len(_data))
train_data, val_data = _data[:_n], _data[_n:]

BLOCK, BATCH = 128, 64


def get_batch(split, gen):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,), generator=gen)
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return x.to(device), y.to(device)


def autocast_ctx():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


def train_config(overrides, seed=0, max_iters=3000, eval_iters=200,
                 lr=3e-4, warmup=150, min_lr=3e-5, wd=0.1):
    torch.manual_seed(seed)                                  # controls model init
    cfg = Config(vocab_size=vocab, block_size=BLOCK, n_layer=4, n_head=4, n_embd=256,
                 dropout=0.1, **overrides)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    train_gen = torch.Generator().manual_seed(7777 + seed)   # identical batch order across configs

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        r = (it - warmup) / max(1, max_iters - warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))

    @torch.no_grad()
    def eval_val():
        model.eval()
        g = torch.Generator().manual_seed(4242)             # identical eval batches for ALL configs
        tot = 0.0
        for _ in range(eval_iters):
            x, y = get_batch("val", g)
            with autocast_ctx():
                _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / eval_iters

    t0 = time.time()
    for it in range(max_iters):
        for grp in opt.param_groups:
            grp["lr"] = get_lr(it)
        x, y = get_batch("train", train_gen)
        with autocast_ctx():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    vl = eval_val()
    return {"val_loss": vl, "val_ppl": math.exp(vl), "params_M": count_params(model) / 1e6,
            "sec": time.time() - t0}
