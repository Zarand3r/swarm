"""Phase 0 data-parallel baseline training loop — the frozen comparison anchor.

A single-worker AdamW run: it wires config → data → model → inner loop → eval →
checkpoint → results.tsv. This loop is intentionally the *inner* loop of DiLoCo
with M=1 and no outer step, so the baseline and DiLoCo's inner block are the same
code.

Determinism: with ``device='cpu'`` and a fixed seed the whole run is
reproducible (the golden-path spine relies on this). On CUDA the forward runs in
bf16 and is gated by loss tolerance, not byte-equality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from swarm.config import RunCfg, config_hash
from swarm.data.loader import ShardedLoader
from swarm.eval import harness
from swarm.metrics import logger
from swarm.model.gpt import GPT
from swarm.train.inner import inner_step


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline(
    cfg: RunCfg,
    *,
    device: str = "cpu",
    max_steps: int | None = None,
    eval_iters: int = 50,
    log_jsonl: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Train the baseline; return a summary dict. Writes nothing unless out_dir given."""
    seed_everything(cfg.seed)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    model = GPT(cfg.model).to(device)
    opt = model.configure_optimizers(
        cfg.optim.weight_decay, cfg.optim.inner_lr,
        (cfg.optim.beta1, cfg.optim.beta2), device_type,
    )

    tokens_per_step = cfg.batch_size * cfg.model.block_size
    total_steps = max_steps if max_steps is not None else max(cfg.token_budget // tokens_per_step, 1)

    loader = ShardedLoader(
        cfg.data_dir, "train", rank=0, world_size=1,
        batch_size=cfg.batch_size, block_size=cfg.model.block_size,
        seed=cfg.seed, device=device,
    )

    model.train()
    for step in range(total_steps):
        x, y = loader.get_batch(step=step)
        loss, gnorm = inner_step(
            model, opt, x, y, opt_cfg=cfg.optim, precision=cfg.precision,
            step=step, total_steps=total_steps, device_type=device_type,
        )
        if log_jsonl is not None and (step % 50 == 0 or step == total_steps - 1):
            logger.append_jsonl({"step": step, "loss": loss, "grad_norm": gnorm}, log_jsonl)

    metrics = harness.estimate_loss(
        model, cfg.data_dir, "val",
        batch_size=cfg.batch_size, block_size=cfg.model.block_size,
        eval_iters=eval_iters, device=device,
    )

    chash = config_hash(cfg)
    summary = {
        "config_hash": chash,
        "seed": cfg.seed,
        "phase": "baseline",
        "steps": total_steps,
        "tokens": total_steps * tokens_per_step,
        "eval_loss": metrics["loss"],
        "eval_bpb": metrics["bpb"],
        "eval_harness_version": harness.EVAL_HARNESS_VERSION,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        from swarm import checkpoint

        checkpoint.save(
            checkpoint.make_state(model, opt, step=total_steps, config_hash=chash),
            out_dir / "checkpoints" / f"baseline_{chash}.pt",
        )
        logger.append_row(
            {
                "config_hash": chash, "seed": cfg.seed, "phase": "baseline",
                "M": cfg.diloco.M, "H": cfg.diloco.H,
                "eval_loss": round(metrics["loss"], 6), "eval_bpb": round(metrics["bpb"], 6),
            },
            path=out_dir / "results.tsv",
        )
    return summary
