#!/usr/bin/env python
"""Phase 0 baseline run.

    python scripts/train_baseline.py --config configs/baseline.yaml --device cuda
    python scripts/train_baseline.py --config configs/baseline.yaml --max-steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from swarm.config import load_yaml
from swarm.train.baseline import train_baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None, help="override token-budget step count")
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    summary = train_baseline(
        cfg, device=args.device, max_steps=args.max_steps,
        eval_iters=args.eval_iters,
        log_jsonl=args.out_dir / "logs" / f"baseline_{cfg.seed}.jsonl",
        out_dir=args.out_dir,
    )
    print(
        f"[baseline] hash={summary['config_hash']} steps={summary['steps']} "
        f"tokens={summary['tokens']:,} eval_loss={summary['eval_loss']:.4f} "
        f"eval_bpb={summary['eval_bpb']:.4f}"
    )


if __name__ == "__main__":
    main()
