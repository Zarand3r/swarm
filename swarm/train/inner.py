"""The AdamW inner training step — Phase 0's loop and DiLoCo's inner block.

One function, ``inner_step``, does exactly one optimizer step: set LR from the
schedule, forward (under bf16 autocast on CUDA), backward, clip, step, zero. It
is reused verbatim as DiLoCo's inner loop, so it carries no DiLoCo-specific
state.

Precision policy (P2): bf16 autocast applies on **CUDA only**; CPU runs fp32.
There is deliberately **no gradient scaler** — bf16 doesn't need one, which
sidesteps the "never scale the pseudo-gradient" hazard entirely. (An fp16 +
scaler path is deferred to Step 22, and even then the scaler stays confined
here, to the inner step.)
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from swarm.config import OptimCfg


def lr_at(step: int, oc: OptimCfg, total_steps: int) -> float:
    """Linear warmup to ``inner_lr``, then cosine decay to ``min_lr`` (or constant)."""
    if oc.warmup_steps > 0 and step < oc.warmup_steps:
        return oc.inner_lr * (step + 1) / oc.warmup_steps
    if oc.lr_schedule == "constant":
        return oc.inner_lr
    # cosine
    if step >= total_steps:
        return oc.min_lr
    denom = max(total_steps - oc.warmup_steps, 1)
    decay_ratio = (step - oc.warmup_steps) / denom
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return oc.min_lr + coeff * (oc.inner_lr - oc.min_lr)


def autocast_ctx(device_type: str, precision: str):
    """bf16 autocast on CUDA; fp32 (nullcontext) on CPU or when precision=fp32."""
    if device_type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cuda" and precision == "fp16":
        # fp16 forward is allowed; the (deferred) scaler would live here, not in
        # the pseudo-gradient path.
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def inner_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    opt_cfg: OptimCfg,
    precision: str,
    step: int,
    total_steps: int,
    device_type: str = "cpu",
) -> tuple[float, float]:
    """Run one training step. Returns ``(loss, grad_norm)`` (pre-clip grad norm)."""
    lr = lr_at(step, opt_cfg, total_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr

    with autocast_ctx(device_type, precision):
        _, loss = model(x, y)

    loss.backward()
    if opt_cfg.grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
    else:
        grad_norm = torch.tensor(0.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item()), float(grad_norm)
