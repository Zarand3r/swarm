"""Tokenize a text corpus into flat ``uint16`` ``.bin`` shards + ``meta.json``.

à la nanoGPT: a memory-mapped flat token array is the substrate for the
deterministic rank-sharded loader (Step 3). We use the **tiktoken gpt2** vocab
(50257 < 2**16, so ``uint16`` is lossless) and a contiguous train/val split so
the split is reproducible and the val set is disjoint from train.

Idempotent: a ``meta.json`` records a hash of the source + tokenizer; a re-run
with the same source and existing bins is a no-op. ``force=True`` rewrites.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken

DTYPE = np.uint16


def encode_to_uint16(text: str, tokenizer: str = "gpt2") -> np.ndarray:
    enc = tiktoken.get_encoding(tokenizer)
    ids = enc.encode_ordinary(text)  # no special tokens
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and int(arr.max()) >= 2**16:
        raise ValueError(f"token id {int(arr.max())} exceeds uint16; wrong tokenizer?")
    return arr.astype(DTYPE)


def _source_hash(text: str, tokenizer: str) -> str:
    h = hashlib.sha256()
    h.update(tokenizer.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _is_current(out_dir: Path, source_hash: str) -> bool:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    if not (out_dir / "train.bin").exists() or not (out_dir / "val.bin").exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("source_hash") == source_hash


def prepare(
    text: str,
    out_dir: str | Path,
    val_fraction: float = 0.005,
    tokenizer: str = "gpt2",
    force: bool = False,
) -> dict:
    """Tokenize ``text`` and write ``{train,val}.bin`` + ``meta.json`` to ``out_dir``.

    The split is contiguous: the final ``val_fraction`` of tokens becomes val,
    the rest train — deterministic and disjoint. Returns the meta dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_hash(text, tokenizer)

    if not force and _is_current(out_dir, source_hash):
        return json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    ids = encode_to_uint16(text, tokenizer)
    n = ids.size
    if n == 0:
        raise ValueError("corpus tokenized to zero tokens")
    n_val = int(n * val_fraction)
    n_val = min(max(n_val, 1), n - 1)  # at least 1 val and 1 train token
    train_ids = ids[: n - n_val]
    val_ids = ids[n - n_val :]

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    enc = tiktoken.get_encoding(tokenizer)
    meta = {
        "tokenizer": tokenizer,
        "vocab_size": enc.n_vocab,
        "dtype": "uint16",
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "source_hash": source_hash,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def prepare_file(
    input_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.005,
    tokenizer: str = "gpt2",
    force: bool = False,
) -> dict:
    """Read a UTF-8 text file and tokenize it via :func:`prepare`."""
    text = Path(input_path).read_text(encoding="utf-8")
    return prepare(text, out_dir, val_fraction=val_fraction, tokenizer=tokenizer, force=force)
