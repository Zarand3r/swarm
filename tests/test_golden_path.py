"""§A golden path (Step 9): the deterministic CPU spine.

A fixed toy baseline run (tiny model, fixed seed, dropout=0, math attention, on
CPU/fp32) must reach a stamped eval loss within 1e-3, and reproduce itself on a
same-seed re-run. This test runs after every later step; if it goes red, the most
recent step broke the pipeline.

Determinism notes: single-threaded CPU + math attention + zero dropout make this
stable to ~1e-3 across machines (not bit-exact — that's what the tolerance is
for; GPU runs are tolerance-gated too, never byte-equal — see §D1).
"""

import json
from pathlib import Path

import numpy as np
import torch

from swarm.config import RunCfg
from swarm.train.baseline import train_baseline

GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "baseline.json"
TOL = 1e-3


def _make_data(data_dir: Path, vocab=128):
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    for split, n in (("train", 8000), ("val", 2000)):
        arr = rng.integers(0, vocab, size=n, dtype=np.uint16)
        (data_dir / f"{split}.bin").write_bytes(arr.tobytes())


def _toy_cfg(data_dir: Path) -> RunCfg:
    return RunCfg.from_dict(
        {
            "seed": 1337,
            "batch_size": 8,
            "precision": "fp32",
            "data_dir": str(data_dir),
            "model": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 32,
                      "vocab_size": 128, "dropout": 0.0, "bias": False, "attn_impl": "math"},
            "optim": {"inner_lr": 1e-3, "warmup_steps": 10, "lr_schedule": "cosine",
                      "grad_clip": 1.0},
        }
    )


def _run(tmp_path) -> float:
    torch.set_num_threads(1)
    data_dir = tmp_path / "data"
    _make_data(data_dir)
    cfg = _toy_cfg(data_dir)
    summary = train_baseline(cfg, device="cpu", max_steps=50, eval_iters=20)
    return summary["eval_loss"]


def test_golden_baseline_loss(tmp_path):
    loss = _run(tmp_path)
    golden = json.loads(GOLDEN.read_text())
    assert abs(loss - golden["eval_loss"]) < TOL, (
        f"golden drift: got {loss:.6f}, expected {golden['eval_loss']:.6f} "
        f"(tol {TOL}). If this change intentionally alters the trajectory, "
        f"re-stamp {GOLDEN}."
    )


def test_golden_is_reproducible(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert abs(a - b) < 1e-9
