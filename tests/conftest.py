from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def bot():
    """One bot indexed over data/, shared by every test: loading the model is the slow part."""
    from rag import RAGBot  # imported lazily so tests that don't need the model run without it

    bot = RAGBot()
    bot.read_and_embed_data(DATA_DIR)
    return bot


@pytest.fixture
def empty_bot(bot):
    """A bot that shares the loaded model but has its own, empty index."""
    fresh = type(bot).__new__(type(bot))
    fresh.sentence_transformer = bot.sentence_transformer
    fresh.chunker = bot.chunker
    fresh.chunks, fresh.embeddings, fresh.openai_client = [], None, None
    return fresh
