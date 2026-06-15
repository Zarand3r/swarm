"""Step 6: the vendored nanoGPT model — shapes, loss, param count, weight tying."""

import torch

from swarm.config import ModelCfg
from swarm.model.gpt import GPT


def _tiny_cfg(**kw):
    base = dict(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=256, dropout=0.0,
                bias=False, attn_impl="math")
    base.update(kw)
    return ModelCfg(**base)


def test_forward_shapes_and_loss():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # untrained cross-entropy ~ ln(vocab); sanity band
    assert 0 < loss.item() < 20


def test_weight_tying():
    model = GPT(_tiny_cfg())
    # wte and lm_head must share the SAME storage (tied), not just equal values
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_param_count_matches_sum():
    model = GPT(_tiny_cfg())
    total = sum(p.numel() for p in model.parameters())
    assert model.get_num_params(non_embedding=False) == total
    # non-embedding count is strictly smaller (drops position embeddings)
    assert model.get_num_params(non_embedding=True) < total


def test_attn_impl_validated():
    import pytest

    with pytest.raises(ValueError):
        GPT(_tiny_cfg(attn_impl="bogus"))
