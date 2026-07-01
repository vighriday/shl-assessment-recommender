"""Tests for the health probe.

The point of the probe is that it tells the truth, so the tests pin the exact
hard/soft weighting: a missing catalog is unhealthy (hard), a missing or unreachable
language model is only degraded (soft), and the overall status is the worst
component. They also pin the two response shapes: the minimal default body (exactly
``{"status": ...}``, the assignment's health contract) and the rich deep body.
"""

from __future__ import annotations

from shl_recommender.observability.health import HealthProbe, Status


def test_all_healthy_is_ok():
    report = HealthProbe(catalog=[object(), object()], llm_configured=True).check()
    assert report.status is Status.OK
    assert report.is_serving
    # Build identity travels with the report.
    assert report.build.service == "shl-recommender"


def test_missing_catalog_is_unhealthy_and_not_serving():
    report = HealthProbe(catalog=[]).check()
    assert report.status is Status.UNHEALTHY
    assert not report.is_serving
    catalog = next(c for c in report.components if c.name == "catalog")
    assert catalog.status is Status.UNHEALTHY


def test_none_catalog_is_unhealthy():
    report = HealthProbe(catalog=None).check()
    assert report.status is Status.UNHEALTHY


def test_unconfigured_llm_is_degraded():
    report = HealthProbe(catalog=[object()], llm_configured=False).check()
    assert report.status is Status.DEGRADED
    assert report.is_serving  # a soft dependency down still serves
    llm = next(c for c in report.components if c.name == "language_model")
    assert llm.status is Status.DEGRADED


def test_overall_is_worst_component():
    # Hard failure (catalog) must dominate a simultaneous soft failure (llm).
    report = HealthProbe(catalog=[], llm_configured=False).check()
    assert report.status is Status.UNHEALTHY


def test_deep_check_not_run_by_default():
    # Without deep=True the probe must not be called, even if one is supplied.
    calls = []
    probe = HealthProbe(
        catalog=[object()],
        llm_probe=lambda: calls.append(1) or True,
    )
    probe.check()  # shallow
    assert calls == []
    llm = next(c for c in probe.check().components if c.name == "language_model")
    assert llm.detail == "configured"


def test_deep_check_reports_reachable_when_probe_succeeds():
    probe = HealthProbe(catalog=[object()], llm_probe=lambda: True)
    report = probe.check(deep=True)
    llm = next(c for c in report.components if c.name == "language_model")
    assert llm.status is Status.OK
    assert llm.detail == "reachable"


def test_deep_check_degrades_when_probe_fails():
    def boom():
        raise RuntimeError("bad key")

    probe = HealthProbe(catalog=[object()], llm_probe=boom)
    report = probe.check(deep=True)
    llm = next(c for c in report.components if c.name == "language_model")
    assert llm.status is Status.DEGRADED
    assert "unreachable" in llm.detail
    # A failed model probe is soft: the service is still serving.
    assert report.is_serving


def test_default_body_is_only_status():
    # The assignment's health contract: exactly {"status": <value>}, nothing else, so a
    # strict {"status": "ok"} check is never tripped by extra keys.
    report = HealthProbe(catalog=[object()], llm_configured=True).check()
    assert report.as_status() == {"status": "ok"}


def test_deep_body_serialises_to_the_rich_shape():
    report = HealthProbe(catalog=[object()], llm_configured=True).check()
    payload = report.as_dict()
    assert set(payload) == {"status", "build", "components"}
    assert {c["name"] for c in payload["components"]} == {"catalog", "language_model"}
    # Every component reports the three documented keys.
    for component in payload["components"]:
        assert set(component) == {"name", "status", "detail"}
