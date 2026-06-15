"""Step 6: with dropout=0 + math attention, forward is bit-reproducible on CPU.

This is the precondition for the deterministic golden path (§A / Step 9): the
math attention kernel and zero dropout make two forwards byte-identical.
"""

import torch

from swarm.config import ModelCfg
from swarm.model.gpt import GPT


def test_forward_bit_identical_cpu():
    cfg = ModelCfg(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256,
                   dropout=0.0, bias=False, attn_impl="math")
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        a, _ = model(idx, idx)
        b, _ = model(idx, idx)
    assert torch.equal(a, b)
