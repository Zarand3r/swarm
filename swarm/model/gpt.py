"""GPT model — VENDORED from nanoGPT.

Upstream: https://github.com/karpathy/nanoGPT  (MIT, Andrej Karpathy)
Pinned commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291  (see scripts/clone_nanogpt.sh)

Adaptations from upstream `model.py` (kept deliberately minimal):
- driven by ``swarm.config.ModelCfg`` instead of nanoGPT's ``GPTConfig``;
- attention path is an explicit ``cfg.attn_impl`` flag — ``"math"`` (deterministic,
  used for the golden path) or ``"flash"`` (fused SDPA, GPU speed) — replacing
  upstream's silent auto-detect (guardrail #2);
- ``from_pretrained`` dropped (it pulls in `transformers`; we never load GPT-2
  weights — this is a from-scratch trainer);
- the GPU-init/`print` chatter removed.
Unchanged: the transformer math, weight tying (``wte`` ↔ ``lm_head``), scaled
residual init, and the decoupled-decay optimizer grouping.

Do NOT replace this with the nanoGPT speedrun fork: its alternative inner
optimizer changes the *inner* optimizer and confounds the DiLoCo study
(guardrail #1; enforced by the Step 19 guard which greps for those tokens).
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from swarm.config import ModelCfg


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's doesn't support bias=False)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        if cfg.attn_impl not in ("math", "flash"):
            raise ValueError(f"attn_impl must be 'math' or 'flash', got {cfg.attn_impl!r}")
        self.attn_impl = cfg.attn_impl
        if cfg.attn_impl == "flash" and not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError("flash attention requires torch>=2.0 SDPA")
        if cfg.attn_impl == "math":
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer("bias", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.attn_impl == "flash":
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.config = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: token embedding and output projection share storage.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"sequence length {t} exceeds block size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference mini-opt: only project the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with decoupled weight decay: 2D tensors decay, 1D (bias/LN) don't."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay = [p for p in param_dict.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and device_type == "cuda") else dict()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
