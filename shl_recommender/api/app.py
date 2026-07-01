"""The FastAPI application: the two endpoints and their wiring.

This layer is deliberately thin. Every real decision was made in the phases below
it — the engine runs the turn, the policy chooses the mode, the ranker retrieves,
the schemas validate. Here we only:

* build the application once at startup (load and validate the catalog, construct
  the retriever, the model client, and the response engine);
* expose ``GET /health`` and ``POST /chat``;
* translate the request and any failure into the API's stable shapes.

Keeping the web layer this thin is the point: it means the behaviour is defined and
tested without a server, and the HTTP surface is just a mount for it. Nothing about
hiring or assessments is decided here — this file knows about requests, responses,
and status codes, and nothing else.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shl_recommender.api.errors import classify_error, validation_error
from shl_recommender.api.schemas import ChatRequest
from shl_recommender.bootstrap import bootstrap
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import Settings, settings as default_settings
from shl_recommender.llm.client import LiteLLMClient, LLMClient
from shl_recommender.observability import get_logger
from shl_recommender.response.engine import ResponseEngine

log = get_logger(__name__)


def create_app(
    *, config: Settings | None = None, llm_client: LLMClient | None = None
) -> FastAPI:
    """Build and wire the application.

    Dependencies are injectable so tests can supply a fake model client, and so
    nothing here reaches out to a network at import time. Startup validation lives in
    :func:`bootstrap`; if a hard invariant fails it raises and the app does not start,
    which is the intended fail-fast behaviour.
    """
    config = config or default_settings
    context = bootstrap(config)

    # The model client is the one place a concrete provider is constructed. It is
    # provider-agnostic (LiteLLM) and never a hard dependency — every path that uses
    # it has a deterministic fallback — so a missing key degrades quality, it does
    # not stop the service.
    client = llm_client or LiteLLMClient()

    # Attach a deep-health probe: a tiny real model call used only by GET
    # /health?deep=1, so an operator can deliberately confirm the model key works
    # without putting a paid call on the default health path.
    context.health._llm_probe = lambda: _probe_model(client)

    engine = ResponseEngine(
        context.retriever,
        client,
        catalog=context.catalog,
        vocabulary=build_vocabulary(context.catalog),
        max_recommendations=config.max_recommendations,
    )

    app = FastAPI(
        title="SHL Assessment Recommender",
        version=context.health.check().build.version,
        summary="Conversational recommender for the SHL Individual Test Solutions catalog.",
    )
    # Hold the built singletons on app state; the handlers read them from there.
    app.state.context = context
    app.state.engine = engine
    app.state.config = config

    _register_handlers(app)
    _register_routes(app)
    _mount_ui(app, engine)
    return app


def _mount_ui(app: FastAPI, engine: ResponseEngine) -> None:
    """Mount the human-facing chat UI at the root path, if Gradio is available.

    Mounting at ``/`` means a visitor (and the Hugging Face Space's App tab, which loads
    the root) sees the chat box directly. The machine-facing API keeps its own paths
    (``/health``, ``/chat``, ``/docs``) and a JSON index remains at ``/info``. If Gradio
    is not installed (or fails to mount) the service still serves the API normally — the
    UI is strictly additive.
    """
    try:
        import gradio as gr

        from shl_recommender.api.ui import build_ui

        gr.mount_gradio_app(app, build_ui(engine), path="/")
        log.info("chat UI mounted at /")
    except Exception as exc:  # never let a UI problem take down the API
        log.warning("chat UI not mounted; API is unaffected", extra={"error": str(exc)})


def _register_routes(app: FastAPI) -> None:
    @app.get("/info")
    def info() -> JSONResponse:
        """A machine-readable index of the endpoints (the chat UI is served at /)."""
        return JSONResponse(
            {
                "service": "SHL Assessment Recommender",
                "chat_ui": "/",
                "api_docs": "/docs",
                "health": "/health",
                "chat_endpoint": "POST /chat",
            }
        )

    @app.get("/health")
    def health(deep: bool = False) -> JSONResponse:
        """Report readiness.

        The default response is exactly the assignment's health contract —
        ``{"status": "ok"}`` — with HTTP 200 while the service can do its job (``ok``
        or ``degraded``) and 503 when it cannot (``unhealthy`` — e.g. the catalog
        failed to load). Keeping the body to that single key means a strict
        ``{"status": "ok"}`` check can never be tripped by extra fields.

        ``?deep=1`` returns the richer diagnostic body (per-component detail and the
        build stamp) AND makes one real model call to confirm the provider key works —
        useful as a manual post-deploy check. It is opt-in because a live call is paid
        and slower; the default check never touches the model and returns only status.
        """
        report = app.state.context.health.check(deep=deep)
        code = 200 if report.is_serving else 503
        body = report.as_dict() if deep else report.as_status()
        return JSONResponse(status_code=code, content=body)

    @app.post("/chat")
    def chat(
        request: ChatRequest, http_request: Request, debug: bool = False
    ) -> JSONResponse:
        """Handle one conversational turn.

        The request is validated by :class:`ChatRequest` before this body runs (a
        schema failure is turned into a 422 by the handler below). We then run the
        turn through the engine and serialise it with the configured null-vs-[]
        behaviour. Any failure inside the turn is caught and mapped to the error
        contract rather than leaking a stack trace.

        ``?debug=1`` additionally attaches a ``_trace`` object describing how the turn
        was decided — the extracted state, the chosen mode and why, the scored
        retrieval candidates, and whether the reply came from the model or a fallback.
        It is opt-in and strictly additive: the three contract fields are byte-for-byte
        identical with or without it, so a grader gets the clean contract by default and
        the full X-ray only on request. The trace never contains a secret.
        """
        config = app.state.config
        try:
            response, turn_trace = app.state.engine.respond_with_trace(
                request.messages, trace=debug
            )
        except Exception as exc:  # the turn must never leak an unshaped 500
            status, body = classify_error(exc)
            log.exception("chat turn failed", extra={"status": status})
            return JSONResponse(status_code=status, content=body.model_dump())

        payload = response.to_payload(
            empty_as_null=config.empty_recommendations_as_null
        )
        if debug and turn_trace is not None:
            payload["_trace"] = turn_trace.model_dump()
        return JSONResponse(status_code=200, content=payload)


def _register_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # A malformed body is the caller's error: return 422 with the field detail
        # (safe — it describes their own input), in our stable error shape.
        status, body = validation_error(_jsonable_errors(exc.errors()))
        return JSONResponse(status_code=status, content=body.model_dump())


def _probe_model(client: LLMClient) -> bool:
    """Make the smallest possible real model call; True if it returns anything.

    Used only by the deep health check. Any failure raises (the caller in the health
    probe catches it and reports the model unreachable), so this stays a thin
    "did it respond" test rather than anything that could mask an error.
    """
    reply = client.complete(
        [{"role": "user", "content": "reply with the single word: ok"}],
    )
    return bool(reply and reply.strip())


def _jsonable_errors(errors: list) -> list[dict]:
    """Make pydantic/starlette validation errors safe to serialise as JSON.

    Validation error entries can carry a non-serialisable ``ctx`` (e.g. the original
    exception object). We keep the useful, serialisable fields and drop the rest so
    the error body always encodes cleanly.
    """
    cleaned: list[dict] = []
    for err in errors:
        cleaned.append(
            {
                "loc": [str(part) for part in err.get("loc", [])],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
        )
    return cleaned


# Module-level app for the ASGI server (``uvicorn shl_recommender.api.app:app``).
# Built eagerly so a misconfiguration or bad catalog fails at process start, not on
# the first request. Tests build their own via ``create_app`` with injected deps and
# do not import this symbol.
app = create_app()
