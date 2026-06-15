"""Step 3 (P1): rank shards are disjoint and complete, by pure index arithmetic.

This is the most dangerous failure in the whole system: if two ranks ever read
the same token, quality is silently inflated and every cross-phase comparison is
void. The assignment must be RNG-free and a strict partition of the corpus.
"""

import numpy as np

from swarm.data.loader import iter_shard_indices, shard_bounds


def _make_bin(tmp_path, n):
    arr = np.arange(n, dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    return tmp_path


def test_disjoint_and_complete_partition(tmp_path):
    n = 1000
    _make_bin(tmp_path, n)
    for world_size in (1, 2, 3, 4, 7):
        covered = []
        seen = set()
        for rank in range(world_size):
            idx = set(iter_shard_indices(tmp_path, "train", rank, world_size))
            assert seen.isdisjoint(idx), f"overlap at ws={world_size} rank={rank}"
            seen |= idx
            covered.append(idx)
        # union covers exactly [0, n)
        assert seen == set(range(n)), f"incomplete cover at ws={world_size}"
        # shards differ in size by at most 1 (balanced)
        sizes = [len(c) for c in covered]
        assert max(sizes) - min(sizes) <= 1


def test_assignment_is_seed_independent(tmp_path):
    """iter_shard_indices takes no seed: assignment cannot depend on RNG."""
    import inspect

    params = inspect.signature(iter_shard_indices).parameters
    assert "seed" not in params


def test_shard_bounds_arithmetic():
    # 10 tokens over 3 ranks -> sizes 4,3,3 ; contiguous, no gaps/overlap
    bounds = [shard_bounds(10, r, 3) for r in range(3)]
    assert bounds == [(0, 4), (4, 7), (7, 10)]
