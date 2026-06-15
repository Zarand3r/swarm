"""Step 1 (P5): config is the single source of truth, with a stable hash.

Invariants:
- identical configs hash identically, regardless of how they were built
  (dict key order must not matter) — so the hash is a content identity;
- any field change changes the hash;
- the hash is process-stable (uses hashlib, not Python's salted hash());
- a config round-trips through YAML unchanged (same hash).
"""

import subprocess
import sys

from swarm.config import RunCfg, config_hash, dump_yaml, load_yaml


def test_same_config_same_hash_regardless_of_key_order():
    a = RunCfg.from_dict({"seed": 1, "model": {"n_layer": 4, "n_embd": 128}})
    b = RunCfg.from_dict({"model": {"n_embd": 128, "n_layer": 4}, "seed": 1})
    assert a == b
    assert config_hash(a) == config_hash(b)


def test_field_change_changes_hash():
    base = RunCfg.from_dict({"seed": 1})
    changed_seed = RunCfg.from_dict({"seed": 2})
    changed_nested = RunCfg.from_dict({"seed": 1, "diloco": {"H": 999}})
    assert config_hash(base) != config_hash(changed_seed)
    assert config_hash(base) != config_hash(changed_nested)


def test_hash_is_process_stable():
    """A fresh interpreter must compute the same hash — no salted hash()."""
    code = (
        "from swarm.config import RunCfg, config_hash;"
        "print(config_hash(RunCfg.from_dict({'seed': 7, 'diloco': {'H': 50, 'M': 2}})))"
    )
    h1 = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    h2 = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert h1 == h2
    # and equals the in-process value
    assert h1 == config_hash(RunCfg.from_dict({"seed": 7, "diloco": {"H": 50, "M": 2}}))


def test_yaml_roundtrip(tmp_path):
    cfg = RunCfg.from_dict(
        {"seed": 3, "model": {"n_layer": 6, "n_head": 6, "n_embd": 384}, "diloco": {"H": 100}}
    )
    path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, path)
    loaded = load_yaml(path)
    assert loaded == cfg
    assert config_hash(loaded) == config_hash(cfg)


def test_unknown_field_fails_fast():
    import pytest

    with pytest.raises((TypeError, KeyError)):
        RunCfg.from_dict({"not_a_field": 1})
    with pytest.raises((TypeError, KeyError)):
        RunCfg.from_dict({"model": {"not_a_field": 1}})
