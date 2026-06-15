"""Frozen evaluation harness — the measuring stick every gate reads.

It must be *frozen* after Phase 0: same checkpoint → same number, forever. To
that end eval batches are drawn with a FIXED eval seed (independent of the
training seed), in fp32 with no autocast, over a fixed number of iterations.
``EVAL_HARNESS_VERSION`` is logged with every result; bumping it is the only
sanctioned way to change eval semantics, and it requires re-stamping the
baseline (enforced by the Step 19 guard).

Reported metrics:
- ``loss`` — mean cross-entropy in nats/token;
- ``bits_per_token`` — ``loss / ln 2``;
- ``bpb`` — true bits-per-byte = ``bits_per_token / (bytes per token)``, where
  bytes/token is a fixed property of the (frozen) eval set + tokenizer. Because
  bytes/token is constant, percentage comparisons in bpb and in loss are
  identical — so the gates are equivalent however you read them — but bpb is the
  honest, tokenizer-independent number.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import torch

from swarm.data.loader import ShardedLoader

EVAL_HARNESS_VERSION = "1"

# Eval batches use this fixed seed so the harness is reproducible regardless of
# the run's training seed.
EVAL_SEED = 20240614


@functools.lru_cache(maxsize=8)
def _bytes_per_token(data_dir: str, split: str, tokenizer: str) -> float:
    """Bytes/token of the eval set (decoded once, cached). 1.0 if unavailable."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding(tokenizer)
        toks = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")
        text = enc.decode([int(t) for t in toks])
        n_bytes = len(text.encode("utf-8"))
        return n_bytes / len(toks) if len(toks) else 1.0
    except Exception:
        # Synthetic/random token bins aren't decodable to meaningful text; fall
        # back to 1.0 so bpb == bits_per_token. Real corpora always decode.
        return 1.0


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data_dir: str | Path,
    split: str,
    *,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str = "cpu",
    tokenizer: str = "gpt2",
) -> dict[str, float]:
    """Mean eval loss over ``eval_iters`` fixed batches. Deterministic per checkpoint."""
    was_training = model.training
    model.eval()
    loader = ShardedLoader(
        data_dir, split, rank=0, world_size=1,
        batch_size=batch_size, block_size=block_size, seed=EVAL_SEED, device=device,
    )
    total = 0.0
    for i in range(eval_iters):
        x, y = loader.get_batch(step=i)
        _, loss = model(x, y)  # fp32, no autocast: stable numbers
        total += float(loss.item())
    if was_training:
        model.train()

    mean = total / eval_iters
    bits_per_token = mean / math.log(2)
    bpt_per_byte = _bytes_per_token(str(data_dir), split, tokenizer)
    bpb = bits_per_token / bpt_per_byte
    return {"loss": mean, "bits_per_token": bits_per_token, "bpb": bpb}
