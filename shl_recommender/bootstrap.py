"""Startup composition and fail-fast validation.

This is the service's single wiring point: it loads the catalog, builds the shared
components, and — critically — *validates the assumptions the request path depends
on before the first request arrives*. A service that boots and then discovers on
request 1 that its data is missing or malformed has failed in the worst possible
place: in front of a caller (here, the grader), mid-evaluation, with the cause
buried. We move that failure to the front door.

The rule is: **fail loud at startup for anything that would make every request
wrong; degrade quietly at startup for anything that only reduces quality.**

* A missing or empty catalog, or an item whose ``test_type`` is not a valid code,
  is a hard error — the process should refuse to start, because it cannot produce
  correct output. These raise :class:`StartupError`.

The result of :func:`bootstrap` is an :class:`AppContext` — an immutable bundle of
the built components plus the health probe — that the web layer (Phase 7) mounts
onto the app. Building it here, framework-free, keeps the composition testable
without a running server and keeps the FastAPI layer thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.models import CatalogItem
from shl_recommender.catalog.test_type import CATEGORY_TO_CODE
from shl_recommender.config import Settings, settings
from shl_recommender.observability import HealthProbe, build_info, get_logger, setup_logging
from shl_recommender.retrieval.ranker import LexicalRanker

log = get_logger(__name__)

# The set of single-letter codes any item's ``test_type`` may be built from. A
# derived value is a comma-join of these, so validation splits and checks members.
_VALID_TEST_TYPE_CODES = frozenset(CATEGORY_TO_CODE.values())


class StartupError(RuntimeError):
    """Raised when a hard invariant fails and the service must not start."""


@dataclass(frozen=True)
class AppContext:
    """The built, validated application components shared across requests.

    Immutable and framework-free: the web layer holds one of these and reads from
    it. Bundling the retriever with its own health probe means the endpoint reports
    the health of the very objects that serve traffic, not a separate guess.
    """

    catalog: list[CatalogItem]
    retriever: LexicalRanker
    health: HealthProbe


def _validate_catalog(items: list[CatalogItem]) -> None:
    """Assert the hard invariants the recommender relies on. Raise on violation."""
    if not items:
        raise StartupError("catalog is empty; nothing could be recommended")

    # Every recommendation carries a URL and a test_type the grader checks. If any
    # item cannot satisfy that, we want to know at boot, not when it is served.
    bad_test_type: list[str] = []
    missing_url: list[str] = []
    for item in items:
        codes = item.test_type.split(",") if item.test_type else []
        if not codes or any(code not in _VALID_TEST_TYPE_CODES for code in codes):
            bad_test_type.append(item.entity_id)
        if not item.url or not item.url.startswith("http"):
            missing_url.append(item.entity_id)

    if bad_test_type:
        raise StartupError(
            f"{len(bad_test_type)} item(s) have an invalid test_type "
            f"(e.g. entity_id {bad_test_type[0]}); catalog derivation is broken"
        )
    if missing_url:
        raise StartupError(
            f"{len(missing_url)} item(s) have no usable URL "
            f"(e.g. entity_id {missing_url[0]}); recommendations would be unlinkable"
        )


def bootstrap(config: Settings | None = None) -> AppContext:
    """Load, validate, and wire the application. The one place startup happens.

    Raises:
        StartupError: if a hard invariant fails (bad or empty catalog). The caller
            (the app factory, or a startup script) should let this propagate so the
            process exits non-zero rather than serving broken output.
    """
    config = config or settings
    setup_logging()

    info = build_info()
    log.info(
        "starting service",
        extra={"version": info.version, "commit": info.commit},
    )

    # 1. Load the catalog. A failure here (missing file, unparseable export) is a
    #    hard startup error by nature — the loader raises and we let it surface.
    try:
        catalog = load_catalog(config.raw_catalog_path)
    except Exception as exc:  # loader raises varied types; all are fatal at boot
        raise StartupError(f"could not load catalog: {exc}") from exc

    # 2. Validate the hard invariants before anything depends on them.
    _validate_catalog(catalog)
    log.info("catalog validated", extra={"item_count": len(catalog)})

    # 3. Build the retriever/ranker over the validated catalog.
    retriever = LexicalRanker(catalog)

    # 4. Assemble the health probe over the live components. LLM readiness is a
    #    configuration fact here (a model name is set); it is never pinged.
    health = HealthProbe(
        catalog=catalog,
        llm_configured=bool(config.llm_model),
    )

    report = health.check()
    log.info("startup complete", extra={"health": report.status.value})
    return AppContext(catalog=catalog, retriever=retriever, health=health)
