"""Step 4: full run-state checkpoint round-trips exactly.

Captures model + inner optimizer + step + RNG, with a forward-declared slot for
the outer-momentum state (the outer optimizer lands in Step 11). Resume tests in
Steps 8/11/18 build on this format.
"""

import torch
import torch.nn as nn

from swarm import checkpoint


def _model_and_opt():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # take one step so the optimizer carries real state (exp_avg, etc.)
    x = torch.randn(4, 8)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()
    return model, opt


def test_state_roundtrip(tmp_path):
    model, opt = _model_and_opt()
    path = tmp_path / "ckpt.pt"

    state = checkpoint.make_state(model, opt, step=7, config_hash="deadbeef")
    assert "outer" in state and state["outer"] is None  # forward-declared slot
    checkpoint.save(state, path)

    # fresh model/opt, then load
    model2 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded = checkpoint.load(path)
    step = checkpoint.apply_state(loaded, model2, opt2, restore_rng=False)

    assert step == 7
    assert loaded["config_hash"] == "deadbeef"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    # optimizer exp_avg state restored
    s1 = opt.state[next(iter(opt.state))]["exp_avg"]
    s2 = opt2.state[next(iter(opt2.state))]["exp_avg"]
    assert torch.allclose(s1, s2)


def test_rng_restore(tmp_path):
    model, opt = _model_and_opt()
    path = tmp_path / "ckpt.pt"

    # snapshot RNG into the checkpoint, then draw the "future" sequence
    state = checkpoint.make_state(model, opt, step=0)
    checkpoint.save(state, path)
    future = torch.rand(5)

    # restoring the checkpoint's RNG must reproduce that same future draw
    checkpoint.apply_state(checkpoint.load(path), model, opt, restore_rng=True)
    assert torch.equal(torch.rand(5), future)
