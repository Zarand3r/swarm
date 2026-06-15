"""Configuration: one dataclass tree is the single source of truth for a run.

Every run logs its fully-resolved config plus a content hash (``config_hash``)
and seed. The hash is a *content identity*: two configs that are field-for-field
equal hash equally regardless of how they were constructed, and any change flips
it. It uses ``hashlib`` (not Python's salted ``hash()``) so it is stable across
processes — a hard requirement for comparing runs logged at different times.

Unknown fields fail fast (typo guard) rather than being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(eq=True)
class ModelCfg:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 50304  # GPT-2 (50257) padded to a multiple of 64
    dropout: float = 0.0  # 0 by default: deterministic golden path (guardrail #2)
    bias: bool = False
    attn_impl: str = "math"  # "math" (deterministic) | "flash" (GPU speed)


@dataclass(eq=True)
class OptimCfg:
    inner_lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    min_lr: float = 6e-5


@dataclass(eq=True)
class DiLoCoCfg:
    H: int = 50  # inner steps per outer round
    M: int = 1  # number of workers (M=1 == Lookahead sanity check)
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True


@dataclass(eq=True)
class RunCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    diloco: DiLoCoCfg = field(default_factory=DiLoCoCfg)
    seed: int = 1337
    token_budget: int = 1_000_000  # total tokens this run trains on
    eval_every: int = 2000  # eval cadence, in inner steps
    batch_size: int = 16
    data_dir: str = "data/tinystories"
    out_dir: str = "results"
    precision: str = "bf16"  # "bf16" (default) | "fp16" | "fp32"
    backend: str = "sim"  # "sim" | "gloo"

    # Names of the nested sub-config fields, mapped to their classes.
    _NESTED = {"model": ModelCfg, "optim": OptimCfg, "diloco": DiLoCoCfg}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunCfg":
        """Build a RunCfg from a (possibly partial, possibly nested) dict.

        Missing fields take their dataclass defaults. Unknown fields raise.
        """
        d = dict(d)  # shallow copy; we pop nested keys
        kwargs: dict[str, Any] = {}
        for name, subcls in cls._NESTED.items():
            if name in d:
                sub = d.pop(name)
                if not isinstance(sub, dict):
                    raise TypeError(f"config field {name!r} must be a mapping, got {type(sub)}")
                kwargs[name] = _build(subcls, sub)
        top_fields = {f.name for f in fields(cls) if f.name not in cls._NESTED}
        unknown = set(d) - top_fields
        if unknown:
            raise KeyError(f"unknown config fields: {sorted(unknown)}")
        kwargs.update(d)
        return cls(**kwargs)


def _build(subcls: type, d: dict[str, Any]):
    valid = {f.name for f in fields(subcls)}
    unknown = set(d) - valid
    if unknown:
        raise KeyError(f"unknown {subcls.__name__} fields: {sorted(unknown)}")
    return subcls(**d)


def to_dict(cfg: RunCfg) -> dict[str, Any]:
    """Plain nested dict of the resolved config (private keys dropped)."""
    d = asdict(cfg)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def config_hash(cfg: RunCfg, length: int = 16) -> str:
    """Stable content hash of a resolved config.

    Canonicalizes via ``json.dumps(..., sort_keys=True)`` so key order is
    irrelevant, then SHA-256. Deterministic across processes.
    """
    canonical = json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def dump_yaml(cfg: RunCfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=True), encoding="utf-8")


def load_yaml(path: str | Path) -> RunCfg:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RunCfg.from_dict(raw)
