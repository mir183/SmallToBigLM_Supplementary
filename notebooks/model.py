"""Configurable decoder-only Transformer for the lever-comparison study.

Every architectural choice is a flag on Config, so baseline vs a variant is a
single change. Levers:
  pos_encoding : sinusoidal (2017) | learned | rope | alibi | none
  norm_placement: post (2017) | pre
  norm_type    : layernorm | rmsnorm
  activation   : relu (2017) | gelu | swiglu
  attention    : mha (2017) | gqa | sliding
Novel tweaks (default off), tested on top of the "modern combo":
  qk_norm      : RMSNorm on Q,K (per head) before attention
  layerscale   : learnable per-channel residual scaling
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    # ---- levers (vanilla 2017 defaults) ----
    pos_encoding: str = "sinusoidal"   # sinusoidal | learned | rope | alibi | none
    norm_placement: str = "post"       # post | pre
    norm_type: str = "layernorm"       # layernorm | rmsnorm
    activation: str = "relu"           # relu | gelu | swiglu
    attention: str = "mha"             # mha | gqa | sliding | diff (Differential Transformer)
    n_kv_head: int = 2                 # used when attention == "gqa"
    window_size: int = 64              # used when attention == "sliding"
    rope_base: float = 10000.0         # RoPE base frequency (larger -> longer-range extrapolation)
    # ---- novel tweaks (default off) ----
    qk_norm: bool = False              # RMSNorm on Q,K per head before attention
    layerscale: bool = False           # learnable per-channel residual scaling
    layerscale_init: float = 0.1
    sandwich_norm: bool = False        # extra post-norm on each sublayer (Gemma2-style, pre-LN)
    logit_softcap: float = 0.0         # tanh soft-cap on final logits (0 = off)
    # ---- efficiency exploration ----
    ffn_mult: float = 4.0              # FFN expansion factor (capacity knob)
    n_unique_layers: int = 0           # 0 = all layers unique; k>0 = cross-layer sharing (k unique blocks)
    conv_kernel: int = 0               # 0 = off; k>0 = depthwise causal conv sub-block (kernel k)
    conv_gate: bool = False            # zero-init per-channel gate on the conv sub-block (ReZero-style; "scale-safe" twist)
    mtp_weight: float = 0.0            # 0 = off; >0 = multi-token prediction: auxiliary t+2 loss
                                       # (light proj + shared tied head, DeepSeek-style). TRAINING ONLY:
                                       # eval loss stays pure next-token CE so ppl is comparable.
    grad_ckpt: bool = False            # activation checkpointing on blocks (training only) - identical
                                       # math/protocol, ~40% less activation VRAM, ~30% slower; lets
                                       # ~163M models train at batch 16 on the 8GB card


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm_type == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def sinusoidal_pe(block_size, n_embd):
    pe = torch.zeros(block_size, n_embd)
    pos = torch.arange(0, block_size).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def alibi_slopes(n_head):
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_head).is_integer():
        return pow2(n_head)
    closest = 2 ** math.floor(math.log2(n_head))
    return pow2(closest) + pow2(2 * closest)[0::2][: n_head - closest]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd) ; cos/sin: (T, hd)
    T = x.size(2)
    cos = cos[:T].to(x.dtype).view(1, 1, T, -1)
    sin = sin[:T].to(x.dtype).view(1, 1, T, -1)
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.diff = (cfg.attention == "diff")
        self.gd = self.hd // 2 if self.diff else self.hd   # per-group dim; diff splits each head into 2 half-width maps
        self.n_kv = cfg.n_kv_head if cfg.attention == "gqa" else cfg.n_head
        assert self.n_head % self.n_kv == 0

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.hd)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * self.n_kv * self.hd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

        T = cfg.block_size
        mask = torch.tril(torch.ones(T, T))
        if cfg.attention == "sliding":
            idx = torch.arange(T)
            band = (idx.view(1, T) > (idx.view(T, 1) - cfg.window_size)).float()
            mask = mask * band
        self.register_buffer("mask", mask.view(1, 1, T, T))

        # positional mechanisms (rope / alibi can be combined for the hybrid)
        self.use_rope = cfg.pos_encoding in ("rope", "rope_alibi")
        self.use_alibi = cfg.pos_encoding in ("alibi", "alibi_learned", "rope_alibi")
        if self.use_rope:
            assert self.gd % 2 == 0
            inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, self.gd, 2).float() / self.gd))
            freqs = torch.outer(torch.arange(T).float(), inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("rope_cos", emb.cos())
            self.register_buffer("rope_sin", emb.sin())
        if self.use_alibi:
            i = torch.arange(T).view(T, 1)
            j = torch.arange(T).view(1, T)
            self.register_buffer("alibi_dist", (i - j).clamp(min=0).float())  # (T,T)
            slopes0 = torch.tensor(alibi_slopes(cfg.n_head)).float()
            if cfg.pos_encoding == "alibi_learned":
                self.alibi_slopes = nn.Parameter(slopes0)          # trainable per-head slopes
            else:
                self.register_buffer("alibi_slopes", slopes0)      # fixed geometric slopes

        if self.diff:  # Differential Transformer (Ye et al. 2024): two softmax maps, subtracted
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)   # depth-dependent (layer_idx 0-based)
            self.lq1 = nn.Parameter(torch.randn(self.gd) * 0.1)
            self.lk1 = nn.Parameter(torch.randn(self.gd) * 0.1)
            self.lq2 = nn.Parameter(torch.randn(self.gd) * 0.1)
            self.lk2 = nn.Parameter(torch.randn(self.gd) * 0.1)
            self.head_norm = RMSNorm(self.hd)                          # per-head output normalization

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv, self.hd)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        if self.cfg.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.diff:
            # split each head into two half-width groups -> two attention maps, subtracted (noise cancelling)
            q1, q2 = q[..., :self.gd], q[..., self.gd:]
            k1, k2 = k[..., :self.gd], k[..., self.gd:]
            if self.use_rope:
                q1, q2 = apply_rope(q1, self.rope_cos, self.rope_sin), apply_rope(q2, self.rope_cos, self.rope_sin)
                k1, k2 = apply_rope(k1, self.rope_cos, self.rope_sin), apply_rope(k2, self.rope_cos, self.rope_sin)
            scale = 1.0 / math.sqrt(self.gd)
            m = self.mask[:, :, :T, :T] == 0
            a1 = self.attn_drop(F.softmax(((q1 @ k1.transpose(-2, -1)) * scale).masked_fill(m, float("-inf")), dim=-1))
            a2 = self.attn_drop(F.softmax(((q2 @ k2.transpose(-2, -1)) * scale).masked_fill(m, float("-inf")), dim=-1))
            lam = torch.exp(torch.dot(self.lq1, self.lk1)) - torch.exp(torch.dot(self.lq2, self.lk2)) + self.lambda_init
            y = (a1 - lam * a2) @ v                                  # (B, nh, T, hd)
            y = self.head_norm(y) * (1.0 - self.lambda_init)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_drop(self.proj(y))

        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
        if self.use_alibi:
            bias = -self.alibi_slopes.view(1, self.n_head, 1, 1) * self.alibi_dist[:T, :T].view(1, 1, T, T)
            att = att + bias.to(att.dtype)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.swiglu = (cfg.activation == "swiglu")
        if self.swiglu:
            hidden = int(cfg.ffn_mult * cfg.n_embd * 2 / 3)
            self.w_gate = nn.Linear(cfg.n_embd, hidden)
            self.w_up = nn.Linear(cfg.n_embd, hidden)
            self.w_down = nn.Linear(hidden, cfg.n_embd)
        else:
            act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
            h = int(cfg.ffn_mult * cfg.n_embd)
            self.net = nn.Sequential(
                nn.Linear(cfg.n_embd, h), act,
                nn.Linear(h, cfg.n_embd),
            )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.swiglu:
            return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
        return self.drop(self.net(x))


class DepthwiseCausalConv(nn.Module):
    """Per-channel causal 1D convolution over the sequence (local token mixing)."""
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(dim, dim, k, groups=dim)

    def forward(self, x):           # x: (B, T, C)
        xt = x.transpose(1, 2)      # (B, C, T)
        xt = F.pad(xt, (self.k - 1, 0))   # left-pad -> causal
        xt = self.conv(xt)
        return xt.transpose(1, 2)


class Block(nn.Module):
    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.cfg = cfg
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ff = MLP(cfg)
        self.n1 = make_norm(cfg)
        self.n2 = make_norm(cfg)
        if cfg.layerscale:
            self.ls1 = nn.Parameter(torch.ones(cfg.n_embd) * cfg.layerscale_init)
            self.ls2 = nn.Parameter(torch.ones(cfg.n_embd) * cfg.layerscale_init)
        if cfg.sandwich_norm:
            self.pn1 = make_norm(cfg)
            self.pn2 = make_norm(cfg)
        if cfg.conv_kernel > 0:
            self.conv = DepthwiseCausalConv(cfg.n_embd, cfg.conv_kernel)
            self.nc = make_norm(cfg)
            if cfg.conv_gate:
                self.conv_g = nn.Parameter(torch.zeros(cfg.n_embd))  # zero-init -> conv starts as a no-op

    def forward(self, x):
        ls1 = self.ls1 if self.cfg.layerscale else 1.0
        ls2 = self.ls2 if self.cfg.layerscale else 1.0
        if self.cfg.conv_kernel > 0:
            cg = self.conv_g if self.cfg.conv_gate else 1.0
            if self.cfg.norm_placement == "pre":
                x = x + cg * self.conv(self.nc(x))
            else:
                x = self.nc(x + cg * self.conv(x))
        if self.cfg.norm_placement == "pre":
            a = self.attn(self.n1(x))
            if self.cfg.sandwich_norm:
                a = self.pn1(a)
            x = x + ls1 * a
            f = self.ff(self.n2(x))
            if self.cfg.sandwich_norm:
                f = self.pn2(f)
            x = x + ls2 * f
        else:  # post-LN (2017)
            x = self.n1(x + ls1 * self.attn(x))
            x = self.n2(x + ls2 * self.ff(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        elif cfg.pos_encoding == "sinusoidal":
            self.register_buffer("pos_emb_sin", sinusoidal_pe(cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        n_uniq = cfg.n_unique_layers if (0 < cfg.n_unique_layers <= cfg.n_layer) else cfg.n_layer
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(n_uniq)])
        self.layer_map = [i % n_uniq for i in range(cfg.n_layer)]  # cross-layer weight sharing
        self.norm_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight   # weight tying
        if cfg.mtp_weight > 0:
            self.mtp_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.mtp_norm = make_norm(cfg)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.cfg.pos_encoding == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos_encoding == "sinusoidal":
            x = x * math.sqrt(self.cfg.n_embd) + self.pos_emb_sin[:T]  # 2017 paper scales embeddings by sqrt(d_model)
        x = self.drop(x)
        for li in self.layer_map:
            if self.cfg.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(self.blocks[li], x, use_reentrant=False)
            else:
                x = self.blocks[li](x)
        hN = self.norm_f(x)
        logits = self.head(hN)
        if self.cfg.logit_softcap and self.cfg.logit_softcap > 0:
            sc = self.cfg.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.training and self.cfg.mtp_weight > 0 and T > 1:
                # auxiliary next-next-token loss: position i also predicts token i+2
                # (targets[i] is token i+1, so token i+2 lives at targets[:, 1:])
                logits2 = self.head(self.mtp_norm(self.mtp_proj(hN)))
                loss2 = F.cross_entropy(logits2[:, :-1].reshape(-1, logits2.size(-1)),
                                        targets[:, 1:].reshape(-1))
                loss = loss + self.cfg.mtp_weight * loss2
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())
