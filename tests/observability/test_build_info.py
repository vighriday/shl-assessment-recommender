"""Tests for build identity resolution.

The resolver must never raise and must fall back cleanly across the three run
environments (installed, source tree, deploy host). We check the source-tree path
resolves a real version, that a commit env var is picked up and shortened, and that
absence degrades to ``"unknown"`` rather than failing.
"""

from __future__ import annotations

from shl_recommender.observability import build_info
from shl_recommender.observability.build_info import (
    BuildInfo,
    _COMMIT_ENV_VARS,
    _resolve,
    _version_from_pyproject,
)


def _clear_cache():
    _resolve.cache_clear()


def test_version_resolves_from_pyproject():
    # Whether or not the package is installed, the source tree carries a version.
    assert _version_from_pyproject() is not None


def test_build_info_shape_and_service_name():
    _clear_cache()
    info = build_info()
    assert isinstance(info, BuildInfo)
    assert info.service == "shl-recommender"
    assert info.version  # non-empty
    assert set(info.as_dict()) == {"service", "version", "commit"}


def test_commit_read_from_env_and_shortened(monkeypatch):
    _clear_cache()
    long_sha = "abcdef0123456789abcdef"
    monkeypatch.setenv(_COMMIT_ENV_VARS[0], long_sha)
    info = build_info()
    assert info.commit == long_sha[:12]
    _clear_cache()


def test_commit_unknown_when_absent(monkeypatch):
    _clear_cache()
    for var in _COMMIT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert build_info().commit == "unknown"
    _clear_cache()
