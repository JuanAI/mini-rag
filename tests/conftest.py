import os
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
DATA_DIR = ROOT / "data"

# Tests never send traces, even when .env enables Langfuse or an OTLP endpoint:
# load_dotenv() doesn't override variables that are already set.
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)


@pytest.fixture(scope="session")
def model():
    """The real embedding model, loaded once per test run: loading it is the slow part."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(MODEL_DIR))


@pytest.fixture(scope="session")
def bot():
    """A RAGBot indexed over data/, shared by every test.

    RAGBot creates its OpenAI client on construction, so a dummy key is set.
    No request is ever sent: tests replace bot.openai_client with a mock.
    """
    from rag import RAGBot

    env = {"OPENAI_API_KEY": "test-key"}
    with patch.dict(os.environ, env):
        os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
        bot = RAGBot()
    bot.read_and_embed_data(DATA_DIR)
    return bot
