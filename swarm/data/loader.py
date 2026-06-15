"""Deterministic, rank-sharded data loading over a memory-mapped token array.

Sharding is the system's most safety-critical contract (P1): each rank owns a
**contiguous, disjoint** block of the token stream, assigned by pure index
arithmetic — *no RNG in assignment*. Overlapping shards would silently inflate
quality and void every comparison, so the partition is trivially verifiable.

RNG appears only in *batch ordering within* a rank's shard, seeded
deterministically by ``(seed, rank, step)`` so a run is reproducible and two
ranks never accidentally correlate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16


def shard_bounds(n_tokens: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return ``[start, end)`` token indices owned by ``rank``.

    A balanced contiguous partition: the first ``n % world_size`` ranks get one
    extra token, so shard sizes differ by at most 1. No RNG, no overlap, full
    cover.
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    base, rem = divmod(n_tokens, world_size)
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return start, start + size


def _bin_path(data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / f"{split}.bin"


def _n_tokens(data_dir: str | Path, split: str) -> int:
    path = _bin_path(data_dir, split)
    return path.stat().st_size // np.dtype(DTYPE).itemsize


def iter_shard_indices(data_dir: str | Path, split: str, rank: int, world_size: int) -> range:
    """The exact token indices owned by ``rank`` — a ``range`` (no RNG, no seed)."""
    n = _n_tokens(data_dir, split)
    start, end = shard_bounds(n, rank, world_size)
    return range(start, end)


class ShardedLoader:
    """Samples ``(x, y)`` next-token batches from one rank's shard.

    A fresh ``np.memmap`` is opened per batch (nanoGPT's trick to avoid a leaking
    mapping); the underlying file is never modified.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        rank: int,
        world_size: int,
        batch_size: int,
        block_size: int,
        seed: int,
        device: str = "cpu",
    ):
        self.path = _bin_path(data_dir, split)
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.seed = seed
        self.device = device

        n = _n_tokens(data_dir, split)
        self.start, self.end = shard_bounds(n, rank, world_size)
        # need block_size+1 tokens for a sequence + its shifted target
        usable = self.end - self.start - (block_size + 1)
        if usable < 0:
            raise ValueError(
                f"shard for rank {rank}/{world_size} has {self.end - self.start} tokens, "
                f"too few for block_size {block_size}"
            )
        self._max_offset = usable  # inclusive upper bound for a start offset

    def _rng(self, step: int) -> np.random.Generator:
        # Deterministic per (seed, rank, step). RNG governs batch *order only*,
        # never which shard — that is fixed by shard_bounds.
        return np.random.default_rng((self.seed, self.rank, step))

    def get_batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.memmap(self.path, dtype=DTYPE, mode="r")
        rng = self._rng(step)
        offsets = rng.integers(0, self._max_offset + 1, size=self.batch_size)
        starts = self.start + offsets

        bs, bk = self.batch_size, self.block_size
        x = np.empty((bs, bk), dtype=np.int64)
        y = np.empty((bs, bk), dtype=np.int64)
        for i, s in enumerate(starts):
            x[i] = data[s : s + bk].astype(np.int64)
            y[i] = data[s + 1 : s + 1 + bk].astype(np.int64)

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device != "cpu":
            xt = xt.to(self.device, non_blocking=True)
            yt = yt.to(self.device, non_blocking=True)
        return xt, yt
