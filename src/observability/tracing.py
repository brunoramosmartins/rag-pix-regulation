"""Phoenix tracing helper — isolates OpenTelemetry/Phoenix dependency."""

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_tracer = None


def _get_tracer():
    """Lazy init tracer. Returns None if Phoenix/OTEL not available."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("rag-pix-regulation", "0.7.0")
        return _tracer
    except ImportError:
        logger.debug("OpenTelemetry not available, tracing disabled")
        return None


def _ensure_phoenix_registered() -> bool:
    """Register Phoenix as trace exporter if not already done."""
    try:
        from phoenix.otel import register
        register(project_name="rag-pix-regulation", auto_instrument=False)
        return True
    except ImportError:
        logger.debug("Phoenix OTEL not available")
        return False
    except Exception as e:
        logger.debug("Phoenix registration failed: %s", e)
        return False


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[None, None, None]:
    """
    Context manager for a traced span.

    If tracing is unavailable, yields without doing anything.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    attrs = attributes or {}
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield
        except Exception as e:
            if span.is_recording():
                try:
                    from opentelemetry import trace as otel_trace
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                except ImportError:
                    pass
            raise
