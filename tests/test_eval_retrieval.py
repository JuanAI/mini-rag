"""Retrieval eval: unit tests for the helpers, plus the CI gate against the committed baseline."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from chunker import Chunk
from eval.checks import invalid_citations
from eval.retrieval import (
    BASELINE_PATH, DATASET_PATH, GATED_METRICS, TOLERANCE,
    evaluate_retrieval, invalid_keywords, load_dataset, norm, rank_of_answer,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def metrics(bot, dataset):
    return evaluate_retrieval(bot, dataset)


class TestNorm:
    @pytest.mark.parametrize("raw, expected", [
        ("£3,000", "3 000"),
        ("MIL-STD-1815", "mil std 1815"),
        ("£3, 000", "3 000"),
        ("  Galileo   GALILEI. ", "galileo galilei"),
    ])
    def test_cases(self, raw, expected):
        assert norm(raw) == expected


class TestDataset:
    def test_shape(self, dataset):
        ids = [item["id"] for item in dataset]
        assert len(ids) == len(set(ids))
        assert {item["answer_type"] for item in dataset} == {"answerable", "unanswerable"}
        answerable = [item for item in dataset if item["answer_type"] == "answerable"]
        assert len(answerable) == 50
        assert all(item["keyword"] and item["source"] for item in answerable)
        assert all("keyword" not in item for item in dataset if item["answer_type"] == "unanswerable")

    def test_every_keyword_exists_in_its_source(self, bot, dataset):
        assert invalid_keywords(bot, dataset) == []


def fake_index():
    """Three chunks on a 2-D unit circle; the question vector is (1, 0)."""
    chunks = [
        Chunk(id=0, text="the answer is 1815", source="data/jupiter.txt"),   # best score, wrong document
        Chunk(id=1, text="born in 1815", source="data/adalovelace.txt"),     # second
        Chunk(id=2, text="unrelated", source="data/adalovelace.txt"),       # last
    ]
    embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    model = SimpleNamespace(encode=lambda *args, **kwargs: np.array([1.0, 0.0]))
    return SimpleNamespace(chunks=chunks, embeddings=embeddings, sentence_transformer=model)


class TestRankOfAnswer:
    def test_rank_skips_chunks_from_other_documents(self):
        item = {"question": "When was Ada born?", "keyword": "1815", "source": "adalovelace"}
        assert rank_of_answer(fake_index(), item) == 2

    def test_none_when_keyword_only_in_another_document(self):
        item = {"question": "q", "keyword": "the answer", "source": "adalovelace"}
        assert rank_of_answer(fake_index(), item) is None


class TestEvaluateRetrieval:
    def test_metrics_are_rates_and_recall_grows_with_k(self, metrics):
        assert all(0 <= value <= 1 for value in metrics.values())
        recalls = [metrics[f"recall@{k}"] for k in (1, 3, 5, 10)]
        assert recalls == sorted(recalls)
        assert {"recall@5/adalovelace", "recall@5/crispr", "recall@5/jupiter"} <= metrics.keys()

    def test_deterministic(self, bot, dataset, metrics):
        assert evaluate_retrieval(bot, dataset) == metrics


def test_retrieval_gate(metrics):
    """CI gate: fails if retrieval gets worse than the committed baseline."""
    baseline = json.loads(BASELINE_PATH.read_text())
    for name in GATED_METRICS:
        assert metrics[name] >= baseline[name] - TOLERANCE, (
            f"{name} dropped to {metrics[name]:.3f} (baseline {baseline[name]:.3f})"
        )


class TestInvalidCitations:
    @pytest.mark.parametrize("answer, allowed, expected", [
        ("Io [3] and Europa [7].", {3, 7}, []),
        ("A [3] B [4]", {3, 7}, [4]),
        ("A [0]", {3}, [0]),
        ("No citations here.", {3}, []),
        ("A [4] B [4] C [9]", {3}, [4, 9]),
        ("Range [2-3] and text [x] are not citations.", {1}, []),
    ])
    def test_cases(self, answer, allowed, expected):
        assert invalid_citations(answer, allowed) == expected
