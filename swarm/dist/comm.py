"""The communication seam — one interface, swappable backends.

``Comm`` is the single abstraction Phase 1 (simulated) and Phase 2 (real
collectives) share, so the DiLoCo driver is written once and only the backend
swaps. Keeping this interface clean is the design's central bet: every deferred
phase (NCCL, gossip, WAN) is just another ``Comm``.

Unified averaging contract: a process contributes the tensors of the workers it
owns; ``all_reduce_mean`` returns the global mean across **all** workers in the
world (total count = ``world_size``). For the simulated backend one process owns
all M workers; for Gloo each process owns one. Same math:

    global_mean = all_reduce_sum( sum(local_contributions) ) / world_size
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

StateDict = dict[str, torch.Tensor]


@runtime_checkable
class Comm(Protocol):
    rank: int
    world_size: int

    def local_worker_ranks(self) -> list[int]:
        """Global indices of the workers this process is responsible for."""
        ...

    def all_reduce_mean(self, contributions: list[StateDict]) -> StateDict:
        """Mean of every worker's state-dict across the whole world (fp32)."""
        ...

    def broadcast_master(self, master: StateDict) -> None:
        """Realign all workers to one authoritative master (in place)."""
        ...

    def barrier(self) -> None:
        ...


def _sum_state_dicts(contributions: list[StateDict]) -> StateDict:
    """Elementwise fp32 sum of a list of like-keyed state-dicts."""
    acc: StateDict = {k: v.detach().clone().float() for k, v in contributions[0].items()}
    for c in contributions[1:]:
        for k in acc:
            acc[k] += c[k].float()
    return acc


class SimComm:
    """In-process backend: one process simulates all ``world_size`` workers.

    Averaging is an exact fp32 mean over the M contributions — the deterministic
    reference the Gloo backend is later checked against (the equivalence gate).
    """

    def __init__(self, world_size: int):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        self.rank = 0
        self.world_size = world_size

    def local_worker_ranks(self) -> list[int]:
        return list(range(self.world_size))

    def all_reduce_mean(self, contributions: list[StateDict]) -> StateDict:
        if len(contributions) != self.world_size:
            raise ValueError(
                f"SimComm owns all {self.world_size} workers but got "
                f"{len(contributions)} contributions"
            )
        local_sum = _sum_state_dicts(contributions)  # all workers are local here
        return {k: v / self.world_size for k, v in local_sum.items()}

    def broadcast_master(self, master: StateDict) -> None:
        # Single source of truth: nothing to send.
        return None

    def barrier(self) -> None:
        return None


class GlooComm:
    """Real ``torch.distributed`` (Gloo) backend — implemented in Step 14."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GlooComm lands in Step 14 (Phase 2 real collectives)")
