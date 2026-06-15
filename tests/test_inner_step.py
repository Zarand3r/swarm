"""Step 7: the AdamW inner training step + LR schedule.

Invariants:
- the LR schedule is linear warmup then cosine decay to min_lr (or constant);
- a step actually learns: repeated steps on one batch drive loss down;
- gradient clipping bounds the update (smaller clip -> smaller param change);
- bf16 path carries NO GradScaler (P2 pre-guard) — confined to inner anyway.
"""

import math
from pathlib import Path

import torch

from swarm.config import ModelCfg, OptimCfg
from swarm.model.gpt import GPT
from swarm.train.inner import inner_step, lr_at


def _model():
    cfg = ModelCfg(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128,
                   dropout=0.0, bias=False, attn_impl="math")
    torch.manual_seed(0)
    return GPT(cfg), cfg


def test_lr_schedule_warmup_then_cosine():
    oc = OptimCfg(inner_lr=1.0, min_lr=0.05, warmup_steps=10, lr_schedule="cosine")
    total = 110
    # warmup is linear: lr = inner_lr * (step+1)/warmup_steps
    assert math.isclose(lr_at(0, oc, total), 0.1, rel_tol=1e-6)
    assert math.isclose(lr_at(9, oc, total), 1.0, rel_tol=1e-6)
    # mid-cosine is strictly between min and max
    mid = lr_at(60, oc, total)
    assert 0.05 < mid < 1.0
    # end of schedule clamps to min_lr
    assert math.isclose(lr_at(total, oc, total), 0.05, rel_tol=1e-6)


def test_lr_schedule_constant():
    oc = OptimCfg(inner_lr=0.5, warmup_steps=0, lr_schedule="constant")
    assert lr_at(0, oc, 100) == 0.5
    assert lr_at(50, oc, 100) == 0.5


def test_overfit_one_batch_decreases_loss():
    model, _ = _model()
    oc = OptimCfg(inner_lr=1e-3, warmup_steps=5, lr_schedule="cosine", grad_clip=1.0)
    opt = model.configure_optimizers(oc.weight_decay, oc.inner_lr, (oc.beta1, oc.beta2), "cpu")
    torch.manual_seed(1)
    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))
    first, _ = inner_step(model, opt, x, y, opt_cfg=oc, precision="fp32", step=0,
                          total_steps=60, device_type="cpu")
    for s in range(1, 60):
        last, _ = inner_step(model, opt, x, y, opt_cfg=oc, precision="fp32", step=s,
                             total_steps=60, device_type="cpu")
    assert last < first * 0.5  # clearly learning the fixed batch


def test_grad_clip_bounds_update():
    oc = OptimCfg(inner_lr=1.0, warmup_steps=0, lr_schedule="constant")
    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))

    def update_norm(grad_clip):
        model, _ = _model()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)  # SGD: update == lr*clipped grad
        before = [p.detach().clone() for p in model.parameters()]
        occ = OptimCfg(inner_lr=1.0, warmup_steps=0, lr_schedule="constant", grad_clip=grad_clip)
        inner_step(model, opt, x, y, opt_cfg=occ, precision="fp32", step=0,
                   total_steps=1, device_type="cpu")
        delta = sum((p.detach() - b).pow(2).sum() for p, b in zip(model.parameters(), before))
        return float(delta.sqrt())

    assert update_norm(0.01) < update_norm(1000.0)


def test_no_gradscaler_in_source():
    src = Path("swarm/train/inner.py").read_text()
    assert "GradScaler" not in src
    assert "autocast" in src  # bf16 path present
