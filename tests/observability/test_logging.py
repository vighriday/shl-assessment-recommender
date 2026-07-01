"""Tests for application logging setup.

Cover the parts we own: the JSON formatter's output shape, promotion of ``extra``
context onto the line, exception rendering, idempotent handler installation, and
that the level from settings is honoured. No log *content* of the wider app is
asserted here — only that the plumbing behaves.
"""

from __future__ import annotations

import json
import logging

from shl_recommender.observability.logging import (
    JsonFormatter,
    get_logger,
    setup_logging,
)


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_has_stable_core_fields():
    record = logging.makeLogRecord(
        {"name": "shl_recommender.x", "levelno": logging.INFO, "levelname": "INFO", "msg": "hello"}
    )
    out = _format(record)
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "shl_recommender.x"
    assert "ts" in out and out["ts"].endswith("+00:00")


def test_json_formatter_promotes_extra_fields():
    # Fields attached via ``extra=`` should appear as top-level keys.
    record = logging.makeLogRecord(
        {"msg": "decided", "levelname": "INFO", "mode": "recommend", "turns": 3}
    )
    out = _format(record)
    assert out["mode"] == "recommend"
    assert out["turns"] == 3


def test_json_formatter_renders_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "shl_recommender", logging.ERROR, __file__, 1, "failed", None, __import__("sys").exc_info()
        )
    out = _format(record)
    assert "exception" in out
    assert "ValueError: boom" in out["exception"]


def test_non_serialisable_extra_degrades_to_string():
    record = logging.makeLogRecord({"msg": "x", "levelname": "INFO", "obj": object()})
    out = _format(record)  # must not raise
    assert isinstance(out["obj"], str)


def test_setup_logging_is_idempotent():
    setup_logging(force=True)
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_shl_handler", False)]
    setup_logging()  # second call must not add another handler
    still_ours = [h for h in root.handlers if getattr(h, "_shl_handler", False)]
    assert len(ours) == 1
    assert len(still_ours) == 1


def test_get_logger_namespaces_under_app_root():
    assert get_logger("shl_recommender.retrieval").name == "shl_recommender.retrieval"
    # A bare module name is placed under the app root.
    assert get_logger("mymod").name == "shl_recommender.mymod"
    # __main__ and empty collapse to the root logger.
    assert get_logger("__main__").name == "shl_recommender"
    assert get_logger(None).name == "shl_recommender"


def test_setup_logging_honours_level(monkeypatch):
    from shl_recommender.observability import logging as logmod

    monkeypatch.setattr(logmod.settings, "log_level", "WARNING")
    setup_logging(force=True)
    assert logging.getLogger().level == logging.WARNING
    # Restore a sane default for other tests.
    monkeypatch.setattr(logmod.settings, "log_level", "INFO")
    setup_logging(force=True)
