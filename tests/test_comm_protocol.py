"""Step 5: the Comm seam. SimComm averages exactly; GlooComm is a guarded stub.

Unified contract (so one driver serves Phase 1 and Phase 2):
- a process contributes its local workers' tensors; all_reduce_mean returns the
  global mean across ALL workers (total = world_size);
- broadcast_master realigns workers to one master (no-op for the single-source
  simulated backend);
- barrier returns.
"""

import pytest
import torch

from swarm.dist.comm import GlooComm, SimComm


def test_sim_local_worker_ranks():
    c = SimComm(world_size=4)
    assert c.rank == 0
    assert c.local_worker_ranks() == [0, 1, 2, 3]


def test_sim_all_reduce_mean_is_exact():
    c = SimComm(world_size=4)
    # four "pseudo-grad" state-dicts with values 1,2,3,4 -> mean 2.5 (exact in fp32)
    contributions = [
        {"w": torch.full((3,), float(v)), "b": torch.tensor([float(v), -float(v)])}
        for v in (1, 2, 3, 4)
    ]
    out = c.all_reduce_mean(contributions)
    assert torch.equal(out["w"], torch.full((3,), 2.5))
    assert torch.equal(out["b"], torch.tensor([2.5, -2.5]))
    assert out["w"].dtype == torch.float32


def test_sim_all_reduce_requires_all_workers():
    c = SimComm(world_size=4)
    with pytest.raises(ValueError):
        c.all_reduce_mean([{"w": torch.ones(2)}])  # only 1 of 4 workers


def test_sim_broadcast_and_barrier_are_noops():
    c = SimComm(world_size=2)
    master = {"w": torch.ones(2)}
    c.broadcast_master(master)  # single source of truth: no change, no error
    assert torch.equal(master["w"], torch.ones(2))
    assert c.barrier() is None


def test_gloo_is_guarded_stub():
    with pytest.raises(NotImplementedError):
        GlooComm()
