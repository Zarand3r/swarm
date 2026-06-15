"""Append-only experiment log.

One row per run in ``results/results.tsv`` plus optional per-step JSONL. The
schema is fixed and shared across every phase (baseline, DiLoCo-sim,
DiLoCo-dist) so a later sweep — or the ``auto-research`` harness — can read it as
a single keyed table.

Design (principal-engineer doctrine: explicit, fail-visible, no silent drops):
- ``COLUMNS`` is the contract. Appending an unknown key is a typo and raises
  ``KeyError`` rather than being silently discarded.
- Unspecified columns are written empty, not omitted — every row is the full
  width, so the TSV stays rectangular and parseable.
- A value containing a tab or newline would corrupt a TSV; that raises
  ``ValueError`` instead of writing a broken file.
- The header is written exactly once, when the file is first created. Appends
  never rewrite it, so the file is safe to append to across many runs/processes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The run-level schema. Order is the on-disk column order; do not reorder
# without migrating existing logs (readers key by name, but humans read by
# position).
COLUMNS: list[str] = [
    "config_hash",
    "seed",
    "phase",
    "M",
    "H",
    "eval_loss",
    "eval_bpb",
    "tok_per_s",
    "bytes_per_sync",
    "comm_ms",
    "compute_ms",
    "verdict",
]

DEFAULT_PATH = Path("results/results.tsv")


def _format(value: Any) -> str:
    """Render a cell, rejecting anything that would break the TSV."""
    s = "" if value is None else str(value)
    if "\t" in s or "\n" in s or "\r" in s:
        raise ValueError(f"TSV cell may not contain tab/newline: {s!r}")
    return s


def append_row(row: dict[str, Any], path: str | os.PathLike = DEFAULT_PATH) -> None:
    """Append one run-row to the TSV at ``path``.

    Keys must be a subset of ``COLUMNS``; unknown keys raise ``KeyError`` and
    nothing is written. Missing columns are emitted empty.
    """
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}; allowed: {COLUMNS}")

    # Format every cell first so a bad value aborts before any file mutation.
    cells = [_format(row.get(col)) for col in COLUMNS]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(cells) + "\n")


def read_rows(path: str | os.PathLike = DEFAULT_PATH) -> list[dict[str, str]]:
    """Read the TSV back as a list of dicts (string values), header-keyed."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    out: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        out.append(dict(zip(header, line.split("\t"))))
    return out


def append_jsonl(record: dict[str, Any], path: str | os.PathLike) -> None:
    """Append one JSON object as a line — for dense per-step traces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
