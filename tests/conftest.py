import pytest

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def model():
    """The real embedding model, loaded once per test run: loading it is the slow part."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)
