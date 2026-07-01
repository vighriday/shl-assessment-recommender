"""Cross-cutting operational concerns: logging, build metadata, health.

Kept in one place so the request path, the startup sequence, and the health
endpoint all report through the same primitives rather than each inventing its
own. Nothing here depends on the web framework, so the pieces can be unit-tested
without standing up a server.
"""

from .build_info import BuildInfo, build_info
from .health import HealthProbe, HealthReport, ComponentHealth
from .logging import get_logger, setup_logging

__all__ = [
    "BuildInfo",
    "build_info",
    "ComponentHealth",
    "get_logger",
    "HealthProbe",
    "HealthReport",
    "setup_logging",
]
