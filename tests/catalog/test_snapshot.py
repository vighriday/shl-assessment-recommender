"""Tests for snapshot save/load and its fidelity to the loaded catalog."""

from __future__ import annotations

import json

import pytest

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.snapshot import (
    SNAPSHOT_VERSION,
    SnapshotError,
    load_snapshot,
    save_snapshot,
)
from shl_recommender.config import settings


@pytest.fixture(scope="module")
def items():
    return load_catalog(settings.raw_catalog_path)


def test_round_trip_is_lossless(tmp_path, items):
    path = tmp_path / "snap.json"
    save_snapshot(items, path)
    restored = load_snapshot(path)
    assert restored == items


def test_saved_snapshot_is_versioned(tmp_path, items):
    path = tmp_path / "snap.json"
    save_snapshot(items, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["version"] == SNAPSHOT_VERSION
    assert envelope["count"] == len(items)


def test_incompatible_version_is_rejected(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"version": 999, "items": []}), encoding="utf-8")
    with pytest.raises(SnapshotError, match="not supported"):
        load_snapshot(path)


def test_corrupt_snapshot_is_rejected(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot(path)


def test_committed_snapshot_matches_raw_catalog():
    # The checked-in snapshot must reflect the current raw export. If this fails,
    # the snapshot is stale and `python -m scripts.build_snapshot` needs re-running.
    if not settings.snapshot_path.exists():
        pytest.skip("snapshot not built yet")
    from_raw = load_catalog(settings.raw_catalog_path)
    from_snapshot = load_snapshot(settings.snapshot_path)
    assert from_snapshot == from_raw
