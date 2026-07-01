"""Read and write the normalised catalog snapshot.

The snapshot is the offline-built artifact the server loads at startup. Building
it once (parse, normalise, derive ``test_type``, tag scope) keeps request-time
startup fast and deterministic, and means the running service never touches the
messy raw export.

The on-disk format is a small JSON envelope: a ``version`` plus the list of
normalised items. The envelope lets the format evolve without silently loading a
stale or incompatible snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import CatalogItem

SNAPSHOT_VERSION = 1


class SnapshotError(Exception):
    """Raised when a snapshot cannot be read or is incompatible."""


def save_snapshot(items: list[CatalogItem], path: Path) -> None:
    """Write ``items`` to ``path`` as a versioned JSON snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "version": SNAPSHOT_VERSION,
        "count": len(items),
        "items": [item.model_dump() for item in items],
    }
    # Pretty-printed and key-sorted so the artifact diffs cleanly in version
    # control and can be inspected by eye.
    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> list[CatalogItem]:
    """Load and validate items from a snapshot written by :func:`save_snapshot`."""
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"could not read snapshot at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"snapshot at {path} is not valid JSON: {exc}") from exc

    version = envelope.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"snapshot version {version} is not supported (expected {SNAPSHOT_VERSION}); rebuild it"
        )

    try:
        return [CatalogItem.model_validate(item) for item in envelope["items"]]
    except KeyError as exc:
        raise SnapshotError(f"snapshot at {path} is missing 'items'") from exc
