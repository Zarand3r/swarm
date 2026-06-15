#!/usr/bin/env python
"""CLI: tokenize a text corpus into {train,val}.bin + meta.json.

Usage:
    # from a local UTF-8 text file
    python scripts/prepare_data.py --input corpus.txt --out-dir data/tinystories

    # idempotent: re-running with the same source is a no-op unless --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarm.data import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="UTF-8 text corpus file")
    ap.add_argument("--out-dir", required=True, type=Path, help="output dir for bins + meta")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--force", action="store_true", help="rewrite even if up to date")
    args = ap.parse_args()

    meta = prepare.prepare_file(
        args.input,
        args.out_dir,
        val_fraction=args.val_fraction,
        tokenizer=args.tokenizer,
        force=args.force,
    )
    print(
        f"[prepare_data] {args.out_dir}: "
        f"train={meta['train_tokens']:,} val={meta['val_tokens']:,} "
        f"vocab={meta['vocab_size']} ({meta['tokenizer']})"
    )


if __name__ == "__main__":
    main()
