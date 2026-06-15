"""Step 0: the append-only results.tsv metrics logger.

Invariants under test:
- a row round-trips: what we append is what we read back;
- the header is written exactly once, no matter how many rows are appended;
- unknown columns fail fast (typo guard) rather than silently dropping data;
- values that would corrupt the TSV (tabs/newlines) fail fast.
"""

import pytest

from swarm.metrics import logger


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "results.tsv"
    row = {"config_hash": "abc123", "seed": 0, "phase": "baseline", "eval_bpb": 1.2345}
    logger.append_row(row, path=path)

    rows = logger.read_rows(path)
    assert len(rows) == 1
    # all columns present; appended values preserved (as strings on read)
    assert rows[0]["config_hash"] == "abc123"
    assert rows[0]["seed"] == "0"
    assert rows[0]["phase"] == "baseline"
    assert rows[0]["eval_bpb"] == "1.2345"
    # unspecified columns are present but empty
    assert rows[0]["bytes_per_sync"] == ""


def test_header_written_once(tmp_path):
    path = tmp_path / "results.tsv"
    logger.append_row({"seed": 0}, path=path)
    logger.append_row({"seed": 1}, path=path)
    logger.append_row({"seed": 2}, path=path)

    text = path.read_text()
    lines = text.splitlines()
    # exactly one header + three data rows
    assert lines[0].split("\t") == logger.COLUMNS
    assert len(lines) == 4
    assert logger.read_rows(path)[2]["seed"] == "2"


def test_unknown_column_fails_fast(tmp_path):
    path = tmp_path / "results.tsv"
    with pytest.raises(KeyError):
        logger.append_row({"not_a_column": 1}, path=path)
    # nothing partial written
    assert not path.exists()


def test_tab_or_newline_value_fails_fast(tmp_path):
    path = tmp_path / "results.tsv"
    with pytest.raises(ValueError):
        logger.append_row({"phase": "a\tb"}, path=path)
    with pytest.raises(ValueError):
        logger.append_row({"phase": "a\nb"}, path=path)


def test_jsonl_append(tmp_path):
    path = tmp_path / "run.jsonl"
    logger.append_jsonl({"step": 1, "loss": 3.0}, path=path)
    logger.append_jsonl({"step": 2, "loss": 2.5}, path=path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    import json

    assert json.loads(lines[1])["loss"] == 2.5
