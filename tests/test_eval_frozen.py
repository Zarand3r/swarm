"""Step 8 (P5): the eval harness is frozen and reproducible.

A frozen, versioned eval is the precondition for every cross-phase gate: the
same checkpoint must always score the same number, and the harness carries a
version so a change can't silently move the baseline.
"""

import numpy as np
import torch

from swarm.config import ModelCfg
from swarm.eval import harness
from swarm.model.gpt import GPT


def _make_val(tmp_path, n=4000, vocab=128):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, vocab, size=n, dtype=np.uint16)
    (tmp_path / "val.bin").write_bytes(arr.tobytes())
    return tmp_path


def test_eval_is_reproducible(tmp_path):
    data_dir = _make_val(tmp_path)
    cfg = ModelCfg(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128,
                   dropout=0.0, bias=False, attn_impl="math")
    torch.manual_seed(0)
    model = GPT(cfg)

    a = harness.estimate_loss(model, data_dir, "val", batch_size=8, block_size=32,
                              eval_iters=10, device="cpu")
    b = harness.estimate_loss(model, data_dir, "val", batch_size=8, block_size=32,
                              eval_iters=10, device="cpu")
    assert a["loss"] == b["loss"]
    assert a["bpb"] == b["bpb"]
    assert a["loss"] > 0 and a["bpb"] > 0


def test_harness_version_present():
    assert isinstance(harness.EVAL_HARNESS_VERSION, str)
    assert harness.EVAL_HARNESS_VERSION
