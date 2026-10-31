"""OpenTelemetry tracing for RAGBot, with the LLM-relevant spans sent to Langfuse.

Off unless configured: without setup_telemetry() the global provider is
OpenTelemetry's no-op one, so RAGBot's spans cost nothing and go nowhere.
A tracing failure never breaks a request.
"""
import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("mini_rag")

# HTTP layers under the OpenAI SDK: noise in Langfuse, and they'd duplicate the LLM span.
HTTP_SCOPES = tuple(
    f"opentelemetry.instrumentation.{name}" for name in ("httpx", "requests", "urllib", "aiohttp")
)
LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")

_provider: TracerProvider | None = None


def build_provider() -> TracerProvider:
    """A provider named after OTEL_SERVICE_NAME. With OTEL_EXPORTER_OTLP_ENDPOINT set,
    every span is also exported over OTLP (Jaeger locally; an APM such as Datadog in production)."""
    provider = TracerProvider(
        resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "mini-rag")})
    )
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def setup_telemetry() -> TracerProvider:
    """Install the global provider and instrument the OpenAI SDK. Calling it again
    returns the same provider."""
    global _provider
    if _provider is None:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        _provider = build_provider()
        trace.set_tracer_provider(_provider)
        OpenAIInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def should_export_to_langfuse(span: ReadableSpan) -> bool:
    """Block the HTTP instrumentation scopes, allow everything else."""
    scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
    return not scope.startswith(HTTP_SCOPES)


def setup_langfuse() -> bool:
    """Attach Langfuse's span processor to the global provider when LANGFUSE_ENABLED=true.
    Returns whether it was attached; a missing key or SDK error only logs a warning."""
    if os.environ.get("LANGFUSE_ENABLED", "").lower() != "true":
        return False
    if missing := [name for name in LANGFUSE_KEYS if not os.environ.get(name)]:
        logger.warning("Langfuse tracing is off: %s not set", ", ".join(missing))
        return False
    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_HOST"],
            tracer_provider=setup_telemetry(),
            should_export_span=should_export_to_langfuse,
        )
    except Exception as exc:  # tracing must never stop the bot
        logger.warning("Langfuse tracing is off: %s", exc)
        return False
    return True


def flush_telemetry() -> None:
    """Send any batched spans now; called when the CLI exits."""
    if _provider is not None:
        _provider.force_flush()
