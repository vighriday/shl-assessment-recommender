"""Application logging.

A deployed service is only debuggable if it says what it is doing — especially
when it degrades. This module gives the rest of the code two things:

* :func:`setup_logging` — called once at startup to install a single formatter on
  the root logger, either machine-readable JSON lines or a compact console form.
* :func:`get_logger` — the accessor every module uses to obtain a named logger.

We build on the standard library's ``logging`` rather than a third-party logging
framework on purpose: it is already a dependency of everything, it needs no extra
package on a memory-constrained host, and structured output is a small formatter
away. The one thing we add is a JSON formatter that promotes any ``extra=`` fields
onto the log line, so a call site can attach context (a turn count, a chosen mode,
a latency) and have it appear as first-class keys a log collector can filter on.

The design rule for the whole service: **log every degradation, never log a
secret.** When the model call fails and we fall back, that is a WARNING with the
reason; when semantic retrieval is unavailable and we run lexical-only, that is a
WARNING at startup. Silent fallbacks are the thing that makes a deployed AI system
impossible to trust, so we do not have them.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any

from shl_recommender.config import settings

# The logger namespace for the whole application. Everything lives under this so a
# deployment can raise or lower the level for our code without touching the noisy
# loggers of the libraries we depend on.
_ROOT_NAME = "shl_recommender"

# Attributes present on every ``LogRecord`` by default. Anything on a record that
# is *not* in this set was attached by a call site via ``extra=`` and is therefore
# structured context we want to surface on the JSON line.
_STANDARD_RECORD_FIELDS = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}

# Set once by ``setup_logging`` so repeated calls (tests, re-imports, a worker that
# re-initialises) do not stack duplicate handlers on the root logger.
_configured = False


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object, one per line.

    The fixed keys (``ts``, ``level``, ``logger``, ``message``) are always present
    so log queries can rely on them; any extra keyword fields the call site passed
    are merged in, and exception information is rendered into a ``exception`` key
    rather than a trailing multi-line traceback that would break line-per-event
    parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # ISO-8601 in UTC with an explicit offset — unambiguous across hosts.
            "ts": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Promote structured context attached via ``extra=`` onto the top level.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # ``default=str`` so an unexpected non-serialisable value degrades to its
        # string form instead of throwing inside the logger.
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        # Compact, readable form for local development. Structured ``extra`` fields
        # are not expanded here — use the JSON format when you need to see them.
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    return handler


def setup_logging(*, force: bool = False) -> None:
    """Install the application's log handler and level. Safe to call more than once.

    Idempotent by default: the first call configures the root logger and later
    calls are no-ops, which is what a normal process wants. ``force=True`` tears
    down and rebuilds the handler — used by tests that switch format mid-run.
    """
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    # Replace any handlers we previously added (or that a host installed) so we do
    # not emit each line twice. We only remove our own handler type to avoid
    # stamping on a handler a test harness deliberately attached.
    for existing in list(root.handlers):
        if getattr(existing, "_shl_handler", False):
            root.removeHandler(existing)

    handler = _build_handler()
    handler._shl_handler = True  # type: ignore[attr-defined]  # tag for safe removal
    root.addHandler(handler)

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the application namespace.

    Pass the module ``__name__`` at the call site; the leading package segment is
    already ``shl_recommender`` so the returned logger sits under the app root and
    inherits its level and handler.
    """
    if not name or name == "__main__":
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
