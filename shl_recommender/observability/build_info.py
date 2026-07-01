"""Build identity for a running instance.

A deployed service should be able to say *which build it is*. Without that, a bug
report or a health check cannot be tied back to a specific version of the code, and
"is the new deploy actually live?" becomes guesswork. This module resolves a small,
stable identity — version, git commit, service name — and exposes it as an
immutable object the health endpoint reports and the logs can reference.

Resolution is defensive on purpose, because the same code runs in three places
with different information available:

* installed as a package  -> version comes from the package metadata;
* run from a source checkout -> version comes from ``pyproject.toml``;
* on a deploy host          -> the platform injects the git SHA as an env var.

Every lookup falls back rather than raising, so build identity never becomes a
reason the service fails to start. Worst case, a field reads ``"unknown"`` — which
is itself useful information.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata

from shl_recommender.config import settings

# Distribution name as declared in ``pyproject.toml`` ([project].name).
_DIST_NAME = "shl-recommender"

# Environment variables a host may set to stamp the deployed commit. We read
# several because the common platforms disagree on the name (Render, Railway,
# generic CI). The first one present wins.
_COMMIT_ENV_VARS = (
    "SHL_GIT_COMMIT",
    "GIT_COMMIT",
    "RENDER_GIT_COMMIT",
    "RAILWAY_GIT_COMMIT_SHA",
    "SOURCE_COMMIT",
)


@dataclass(frozen=True)
class BuildInfo:
    """Immutable identity of the running build."""

    service: str
    version: str
    commit: str

    def as_dict(self) -> dict[str, str]:
        return {"service": self.service, "version": self.version, "commit": self.commit}


def _version_from_metadata() -> str | None:
    try:
        return metadata.version(_DIST_NAME)
    except metadata.PackageNotFoundError:
        return None


def _version_from_pyproject() -> str | None:
    """Read ``[project].version`` from the source tree, for uninstalled runs."""
    pyproject = settings.project_root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    version = data.get("project", {}).get("version")
    return str(version) if version else None


def _resolve_commit() -> str:
    import os

    for var in _COMMIT_ENV_VARS:
        value = os.environ.get(var)
        if value:
            # Short form is enough to identify a build and reads better in a health
            # payload; keep the full SHA only if it is already short.
            return value[:12]
    return "unknown"


@lru_cache(maxsize=1)
def _resolve() -> BuildInfo:
    version = _version_from_metadata() or _version_from_pyproject() or "unknown"
    return BuildInfo(service=_DIST_NAME, version=version, commit=_resolve_commit())


def build_info() -> BuildInfo:
    """Return the running build's identity. Cached; resolved once per process."""
    return _resolve()
