"""Tracing tests: spans are captured in memory, nothing is sent over the network."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langfuse._client.span_processor import LangfuseSpanProcessor
from openai import OpenAIError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import StatusCode

import telemetry
from telemetry import setup_langfuse, setup_telemetry, should_export_to_langfuse

LANGFUSE_KEYS = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
}


@pytest.fixture(scope="session")
def exporter():
    """One in-memory exporter on the global provider; OTel allows no second provider.
    Langfuse's processor is attached too, exporting to memory: it is what copies
    session and user ids onto child spans."""
    exporter = InMemorySpanExporter()
    provider = setup_telemetry()
    provider.add_span_processor(LangfuseSpanProcessor(
        public_key="pk-lf-test", secret_key="sk-lf-test", base_url="http://localhost",
        span_exporter=InMemorySpanExporter(),
    ))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def spans(exporter):
    exporter.clear()
    return exporter


def by_name(exporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def mock_client(answer="Io [3].", usage=(120, 8)) -> Mock:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )
    return client


def span_with_scope(scope: str) -> ReadableSpan:
    return ReadableSpan(name="s", instrumentation_scope=InstrumentationScope(scope))


class TestSetupLangfuse:
    @pytest.mark.parametrize("value", [None, "false", "no"])
    def test_off_unless_enabled(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
        else:
            monkeypatch.setenv("LANGFUSE_ENABLED", value)
        with patch("langfuse.Langfuse") as langfuse_cls:
            assert setup_langfuse() is False
        langfuse_cls.assert_not_called()

    def test_missing_key_warns_and_returns_false(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        for name, value in LANGFUSE_KEYS.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY")
        with patch("langfuse.Langfuse") as langfuse_cls:
            assert setup_langfuse() is False
        langfuse_cls.assert_not_called()
        assert "LANGFUSE_SECRET_KEY" in caplog.text

    def test_sdk_failure_warns_and_returns_false(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        for name, value in LANGFUSE_KEYS.items():
            monkeypatch.setenv(name, value)
        with patch("langfuse.Langfuse", side_effect=RuntimeError("boom")):
            assert setup_langfuse() is False
        assert "boom" in caplog.text

    def test_attaches_to_the_global_provider_with_the_filter(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        for name, value in LANGFUSE_KEYS.items():
            monkeypatch.setenv(name, value)
        with patch("langfuse.Langfuse") as langfuse_cls:
            assert setup_langfuse() is True
        kwargs = langfuse_cls.call_args.kwargs
        assert kwargs["public_key"] == "pk-lf-test"
        assert kwargs["secret_key"] == "sk-lf-test"
        assert kwargs["base_url"] == "https://cloud.langfuse.com"
        assert kwargs["tracer_provider"] is setup_telemetry()
        assert kwargs["should_export_span"] is should_export_to_langfuse


class TestShouldExportToLangfuse:
    @pytest.mark.parametrize("scope", [
        "opentelemetry.instrumentation.httpx",
        "opentelemetry.instrumentation.requests",
        "opentelemetry.instrumentation.urllib",
        "opentelemetry.instrumentation.urllib3",
        "opentelemetry.instrumentation.aiohttp_client",
    ])
    def test_http_layers_are_blocked(self, scope):
        assert should_export_to_langfuse(span_with_scope(scope)) is False

    @pytest.mark.parametrize("scope", ["mini_rag", "opentelemetry.instrumentation.openai.v1"])
    def test_everything_else_is_exported(self, scope):
        assert should_export_to_langfuse(span_with_scope(scope)) is True


class TestSetupTelemetry:
    def test_idempotent_and_named(self, exporter):
        provider = setup_telemetry()
        assert setup_telemetry() is provider
        assert provider.resource.attributes["service.name"] == "mini-rag"

    def test_service_name_from_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "rag-test")
        assert telemetry.build_provider().resource.attributes["service.name"] == "rag-test"

    def test_otlp_exporter_only_when_endpoint_is_set(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert telemetry.build_provider()._active_span_processor._span_processors == ()
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert len(telemetry.build_provider()._active_span_processor._span_processors) == 1


class TestAskSpans:
    def test_one_trace_with_ask_as_root(self, bot, spans, monkeypatch):
        monkeypatch.setattr(bot, "openai_client", mock_client())
        bot.ask("What are Jupiter's moons?", session_id="s-1", user_id="juan")

        found = by_name(spans)
        assert set(found) == {"rag.ask", "rag.retrieve", "rag.generate"}
        root = found["rag.ask"]
        assert root.parent is None
        assert found["rag.retrieve"].parent.span_id == root.context.span_id
        assert found["rag.generate"].parent.span_id == root.context.span_id
        assert len({span.context.trace_id for span in found.values()}) == 1

    def test_attributes(self, bot, spans, monkeypatch):
        monkeypatch.setattr(bot, "openai_client", mock_client(answer="Io [3]."))
        monkeypatch.setenv("MODEL_NAME", "gpt-test")
        bot.ask("What are Jupiter's moons?", k=3, session_id="s-1", user_id="juan")

        found = by_name(spans)
        root = found["rag.ask"].attributes
        assert root["langfuse.observation.input"] == "What are Jupiter's moons?"
        assert root["langfuse.observation.output"] == "Io [3]."
        assert root["session.id"] == "s-1"
        assert root["user.id"] == "juan"
        assert root["rag.k"] == 3

        retrieve = found["rag.retrieve"].attributes
        assert len(retrieve["rag.chunk_ids"]) == len(retrieve["rag.sources"]) == 3
        assert 0 < retrieve["rag.top_score"] <= 1

        generate = found["rag.generate"].attributes
        assert generate["rag.model"] == "gpt-test"
        assert generate["rag.input_tokens"] == 120
        assert generate["rag.output_tokens"] == 8

    def test_session_and_user_are_on_every_span(self, bot, spans, monkeypatch):
        # Langfuse sums cost per session and user over the spans carrying the id.
        monkeypatch.setattr(bot, "openai_client", mock_client())
        bot.ask("What are Jupiter's moons?", session_id="s-1", user_id="juan")

        for name, span in by_name(spans).items():
            assert span.attributes["session.id"] == "s-1", name
            assert span.attributes["user.id"] == "juan", name

    def test_without_ids_there_are_no_session_attributes(self, bot, spans, monkeypatch):
        monkeypatch.setattr(bot, "openai_client", mock_client())
        bot.ask("What are Jupiter's moons?")

        root = by_name(spans)["rag.ask"].attributes
        assert "session.id" not in root and "user.id" not in root

    def test_openai_error_marks_spans_and_still_raises(self, bot, spans, monkeypatch):
        client = Mock()
        client.chat.completions.create.side_effect = OpenAIError("key expired")
        monkeypatch.setattr(bot, "openai_client", client)

        with pytest.raises(Exception, match="Error raised from OpenAI model key expired"):
            bot.ask("What are Jupiter's moons?")

        found = by_name(spans)
        for name in ("rag.ask", "rag.generate"):
            assert found[name].status.status_code == StatusCode.ERROR
            assert any(event.name == "exception" for event in found[name].events)
