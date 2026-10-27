"""RAGBot tests: real local embedding model, mocked OpenAI client."""
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from openai import OpenAIError

import rag
from rag import RAGBot


def mock_client(answer: str = "An answer [0].") -> Mock:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    return client


def sent_messages(client: Mock) -> list[dict]:
    return client.chat.completions.create.call_args.kwargs["messages"]


class TestClientChoice:
    def test_openai_without_azure_endpoint(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env), patch.object(rag, "OpenAI") as openai_cls, \
                patch.object(rag, "AzureOpenAI") as azure_cls, patch.object(rag, "SentenceTransformer"):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            bot = RAGBot()

        openai_cls.assert_called_once_with(api_key="sk-test")
        azure_cls.assert_not_called()
        assert bot.openai_client is openai_cls.return_value

    def test_azure_when_endpoint_is_set(self):
        env = {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "OPENAI_API_VERSION": "2024-10-21",
        }
        with patch.dict(os.environ, env), patch.object(rag, "OpenAI") as openai_cls, \
                patch.object(rag, "AzureOpenAI") as azure_cls, patch.object(rag, "SentenceTransformer"):
            RAGBot()

        azure_cls.assert_called_once_with(
            api_key="az-key",
            api_version="2024-10-21",
            azure_endpoint="https://example.openai.azure.com/",
        )
        openai_cls.assert_not_called()

    def test_loads_the_bundled_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), patch.object(rag, "OpenAI"), \
                patch.object(rag, "SentenceTransformer") as st_cls:
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            RAGBot()
        st_cls.assert_called_once_with("models/all-MiniLM-L6-v2")


class TestIndexing:
    def test_one_unit_vector_per_chunk(self, bot):
        assert len(bot.chunks) == bot.embeddings.shape[0] > 0
        np.testing.assert_allclose(np.linalg.norm(bot.embeddings, axis=1), 1, atol=1e-5)

    def test_every_document_is_indexed(self, bot):
        sources = {os.path.basename(c.source) for c in bot.chunks}
        assert sources == {"adalovelace.txt", "crispr.txt", "jupiter.txt"}

    def test_files_without_a_parser_are_skipped(self, bot, tmp_path):
        (tmp_path / ".DS_Store").write_bytes(b"ignored")
        (tmp_path / "notes.md").write_text("Ignored markdown.")
        (tmp_path / "planet.txt").write_text("Jupiter is a gas giant.")
        index = RAGBot.__new__(RAGBot)
        index.sentence_transformer = bot.sentence_transformer

        index.read_and_embed_data(tmp_path)

        assert [os.path.basename(c.source) for c in index.chunks] == ["planet.txt"]


class TestRetrieval:
    @pytest.mark.parametrize("question, source", [
        ("Who did Ada Lovelace collaborate with?", "adalovelace.txt"),
        ("What are the four Galilean moons?", "jupiter.txt"),
        ("What does CRISPR stand for?", "crispr.txt"),
    ])
    def test_top_chunk_comes_from_the_matching_document(self, bot, question, source):
        bot.openai_client = mock_client()
        bot.ask(question)
        context = sent_messages(bot.openai_client)[1]["content"]
        first_block = context.split("\n\n")[0]
        assert source in first_block

    def test_uses_k_chunks(self, bot):
        bot.openai_client = mock_client()
        bot.ask("Jupiter", k=3)
        assert sent_messages(bot.openai_client)[1]["content"].count("Source: ") == 3

    def test_fewer_chunks_than_k_uses_all_of_them(self, bot, tmp_path):
        (tmp_path / "planet.txt").write_text("Jupiter is a gas giant.")
        small = RAGBot.__new__(RAGBot)
        small.sentence_transformer = bot.sentence_transformer
        small.read_and_embed_data(tmp_path)
        small.openai_client = mock_client()

        small.ask("Jupiter", k=5)

        assert sent_messages(small.openai_client)[1]["content"].count("Source: ") == 1


class TestGeneration:
    def test_request(self, bot):
        bot.openai_client = mock_client()
        with patch.dict(os.environ, {"MODEL_NAME": "my-model"}):
            bot.ask("What does CRISPR stand for?")

        kwargs = bot.openai_client.chat.completions.create.call_args.kwargs
        system, user = kwargs["messages"]
        assert kwargs["model"] == "my-model"
        assert kwargs["temperature"] == 0
        assert system["role"] == "system"
        assert "Answer using only the supplied context. If it does not contain the answer, " \
               "say you don't know based on these documents. " in system["content"]
        assert "Cite supporting passages with [1], [2] etc. Be concise." in system["content"]
        assert user["content"].startswith("Context:\n[")
        assert user["content"].endswith("\n\nQuestion: What does CRISPR stand for?")

    def test_model_defaults_to_gpt_4_1_mini(self, bot):
        bot.openai_client = mock_client()
        with patch.dict(os.environ):
            os.environ.pop("MODEL_NAME", None)
            bot.ask("Jupiter")
        assert bot.openai_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4.1-mini"

    def test_reply_lists_the_sources(self, bot):
        bot.openai_client = mock_client("Charles Babbage [3].")

        reply = bot.ask("Who did Ada Lovelace collaborate with?", k=2)

        answer, sources = reply.split(" \n\nRetrieved sources:\n")
        assert answer == "Charles Babbage [3]."
        lines = sources.split("\n")
        assert len(lines) == 2
        assert all(line.startswith("[chunk id ") for line in lines)

    def test_openai_error_is_raised_with_context(self, bot):
        bot.openai_client = Mock()
        bot.openai_client.chat.completions.create.side_effect = OpenAIError("boom")

        with pytest.raises(Exception, match="Error raised from OpenAI model boom"):
            bot.ask("Jupiter")
