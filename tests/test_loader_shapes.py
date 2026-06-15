"""Step 3: get_batch returns correctly-shaped, shard-confined, deterministic batches."""

import numpy as np
import torch

from swarm.data.loader import ShardedLoader, shard_bounds


def _make_bin(tmp_path, n):
    # token value == index, so a returned token reveals which shard it came from.
    arr = np.arange(n, dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    return tmp_path


def test_shapes_and_shift(tmp_path):
    _make_bin(tmp_path, 2000)
    block_size, batch_size = 16, 8
    ld = ShardedLoader(tmp_path, "train", rank=0, world_size=1,
                       batch_size=batch_size, block_size=block_size, seed=0)
    x, y = ld.get_batch(step=0)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    # y is x shifted by one position (next-token target)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_batches_confined_to_shard(tmp_path):
    n = 4000
    _make_bin(tmp_path, n)
    world_size = 4
    block_size, batch_size = 16, 8
    for rank in range(world_size):
        start, end = shard_bounds(n, rank, world_size)
        ld = ShardedLoader(tmp_path, "train", rank=rank, world_size=world_size,
                           batch_size=batch_size, block_size=block_size, seed=0)
        for step in range(20):
            x, y = ld.get_batch(step=step)
            # token value == source index; every token must lie in this shard
            assert int(x.min()) >= start
            assert int(y.max()) < end


def test_determinism_same_seed_step(tmp_path):
    _make_bin(tmp_path, 2000)
    kw = dict(rank=0, world_size=1, batch_size=8, block_size=16, seed=42)
    a = ShardedLoader(tmp_path, "train", **kw).get_batch(step=5)
    b = ShardedLoader(tmp_path, "train", **kw).get_batch(step=5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # different step -> (almost surely) different batch
    c = ShardedLoader(tmp_path, "train", **kw).get_batch(step=6)
    assert not torch.equal(a[0], c[0])
