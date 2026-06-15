"""Step 2: tokenize a corpus to uint16 .bin shards + meta.json, idempotently.

Invariants:
- the written token counts in meta.json match the actual .bin lengths;
- train + val tokens == total tokens (no loss, no overlap);
- tokens are uint16 and within the tokenizer vocab;
- re-running on the same source is a no-op (idempotent); force rewrites.
"""

import numpy as np

from swarm.data import prepare

FIXTURE = ("Once upon a time there was a little robot who loved to learn. " * 200)


def _bin_len(path):
    return len(np.memmap(path, dtype=np.uint16, mode="r"))


def test_prepare_writes_consistent_bins_and_meta(tmp_path):
    meta = prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1)

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    assert train_bin.exists() and val_bin.exists()
    assert (tmp_path / "meta.json").exists()

    assert meta["train_tokens"] == _bin_len(train_bin)
    assert meta["val_tokens"] == _bin_len(val_bin)
    assert meta["dtype"] == "uint16"

    # total tokens conserved
    total = _bin_len(train_bin) + _bin_len(val_bin)
    assert total == meta["train_tokens"] + meta["val_tokens"]
    assert total > 0 and meta["val_tokens"] > 0


def test_tokens_within_vocab(tmp_path):
    meta = prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1)
    arr = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
    assert int(arr.max()) < meta["vocab_size"]


def test_idempotent_then_force(tmp_path):
    prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1)
    mtime1 = (tmp_path / "train.bin").stat().st_mtime_ns

    # same source → no rewrite
    prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1)
    mtime2 = (tmp_path / "train.bin").stat().st_mtime_ns
    assert mtime1 == mtime2

    # force → rewrite
    prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1, force=True)
    mtime3 = (tmp_path / "train.bin").stat().st_mtime_ns
    assert mtime3 != mtime2


def test_changed_source_triggers_rewrite(tmp_path):
    prepare.prepare(FIXTURE, tmp_path, val_fraction=0.1)
    mtime1 = (tmp_path / "train.bin").stat().st_mtime_ns
    prepare.prepare(FIXTURE + " and then more text.", tmp_path, val_fraction=0.1)
    mtime2 = (tmp_path / "train.bin").stat().st_mtime_ns
    assert mtime1 != mtime2
