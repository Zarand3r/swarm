"""Run-state checkpointing: model + inner optimizer + outer momentum + step + RNG.

One format serves Phase 0 (resume) and the later distributed/outer phases. The
``outer`` slot is forward-declared (``None`` until the outer optimizer exists in
Step 11) so the format never changes shape across phases — a resumed run always
finds the slot it expects.

RNG capture covers torch (CPU + CUDA), numpy's global state, and Python's
``random`` so a resumed run continues the same stochastic stream.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    outer_optimizer: torch.optim.Optimizer | None = None,
    config_hash: str | None = None,
    include_rng: bool = True,
) -> dict[str, Any]:
    """Build a checkpoint dict with explicit, fixed slots."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # forward-declared: the outer (Nesterov) optimizer's momentum state.
        "outer": outer_optimizer.state_dict() if outer_optimizer is not None else None,
        "step": int(step),
        "config_hash": config_hash,
        "rng": capture_rng_state() if include_rng else None,
    }


def apply_state(
    state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> int:
    """Load a checkpoint dict into live objects; return the saved step."""
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if outer_optimizer is not None and state.get("outer") is not None:
        outer_optimizer.load_state_dict(state["outer"])
    if restore_rng and state.get("rng") is not None:
        restore_rng_state(state["rng"])
    return int(state["step"])


def save(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
