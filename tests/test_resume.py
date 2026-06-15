"""Step 8 (P5): training resumed from a checkpoint continues the same curve.

Train N steps, checkpoint, train N more; compare to an uninterrupted 2N-step run
with the same seed. On CPU/fp32 the loader is deterministic by (seed, rank, step)
and optimizer+RNG are restored, so the two must match to tight tolerance.
"""

import numpy as np
import torch

from swarm import checkpoint
from swarm.config import ModelCfg, OptimCfg
from swarm.eval import harness
from swarm.model.gpt import GPT
from swarm.data.loader import ShardedLoader
from swarm.train.inner import inner_step


def _make_data(tmp_path, vocab=128):
    rng = np.random.default_rng(1)
    for split, n in (("train", 8000), ("val", 2000)):
        arr = rng.integers(0, vocab, size=n, dtype=np.uint16)
        (tmp_path / f"{split}.bin").write_bytes(arr.tobytes())
    return tmp_path


def _new_model_opt():
    cfg = ModelCfg(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128,
                   dropout=0.0, bias=False, attn_impl="math")
    torch.manual_seed(0)
    model = GPT(cfg)
    oc = OptimCfg(inner_lr=1e-3, warmup_steps=2, lr_schedule="cosine", grad_clip=1.0)
    opt = model.configure_optimizers(oc.weight_decay, oc.inner_lr, (oc.beta1, oc.beta2), "cpu")
    return model, opt, oc, cfg


def _run(model, opt, oc, data_dir, start, end, total, seed=7):
    ld = ShardedLoader(data_dir, "train", rank=0, world_size=1,
                       batch_size=8, block_size=32, seed=seed)
    for step in range(start, end):
        x, y = ld.get_batch(step=step)
        inner_step(model, opt, x, y, opt_cfg=oc, precision="fp32",
                   step=step, total_steps=total, device_type="cpu")


def _eval(model, data_dir):
    return harness.estimate_loss(model, data_dir, "val", batch_size=8, block_size=32,
                                 eval_iters=10, device="cpu")["loss"]


def test_resume_matches_uninterrupted(tmp_path):
    data_dir = _make_data(tmp_path)
    N, total = 20, 40

    # interrupted: N steps -> checkpoint -> N more
    model, opt, oc, _ = _new_model_opt()
    _run(model, opt, oc, data_dir, 0, N, total)
    ckpt = tmp_path / "ck.pt"
    checkpoint.save(checkpoint.make_state(model, opt, step=N), ckpt)

    model2, opt2, oc2, _ = _new_model_opt()
    step = checkpoint.apply_state(checkpoint.load(ckpt), model2, opt2, restore_rng=True)
    _run(model2, opt2, oc2, data_dir, step, total, total)
    resumed_loss = _eval(model2, data_dir)

    # uninterrupted: 2N steps straight
    model3, opt3, oc3, _ = _new_model_opt()
    _run(model3, opt3, oc3, data_dir, 0, total, total)
    straight_loss = _eval(model3, data_dir)

    assert abs(resumed_loss - straight_loss) < 1e-4
