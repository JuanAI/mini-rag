"""eval/retrieval/langfuse_log.py: scores and what is published, with the Langfuse client mocked."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eval.retrieval.langfuse_log import DATASET_NAME, item_scores, log_to_langfuse, run_scores
from eval.retrieval.metrics import metrics_from_ranks

LANGFUSE_ENV = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_HOST": "http://localhost:3100",
}
ITEMS = [
    {"id": "q01", "question": "When was Ada born?", "keyword": "1815", "source": "adalovelace"},
    {"id": "q02", "question": "What is Io?", "keyword": "io", "source": "jupiter"},
]


class TestLangfuse:
    @pytest.mark.parametrize("rank, expected", [
        (2, {"rank": 2, "hit@5": 1}),
        (7, {"rank": 7, "hit@5": 0}),
        (None, {"hit@5": 0}),          # not found: no rank score
    ])
    def test_item_scores(self, rank, expected):
        scores = item_scores(input={}, output={"rank": rank}, expected_output={}, metadata={})
        assert {score.name: score.value for score in scores} == expected

    def test_run_scores_match_the_console_metrics(self):
        all_ranks = [[1], [3, 4], [], [12]]
        results = [SimpleNamespace(output={"relevant_ranks": ranks}) for ranks in all_ranks]

        scores = {score.name: score.value for score in run_scores(item_results=results)}

        assert scores == pytest.approx(metrics_from_ranks(all_ranks, (1, 3, 5, 10)))
        assert scores["recall@5"] == 0.5 and scores["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 12) / 4)

    def test_publishes_dataset_items_and_one_run(self, monkeypatch):
        for name, value in LANGFUSE_ENV.items():
            monkeypatch.setenv(name, value)
        with patch("langfuse.Langfuse") as langfuse_cls:
            client = langfuse_cls.return_value
            client.create_dataset_item.side_effect = lambda **kwargs: SimpleNamespace(id=kwargs["id"])
            client.run_experiment.return_value = SimpleNamespace(dataset_run_url="http://run")

            url = log_to_langfuse("chunk-80", {"chunk_size": 80}, ITEMS, [[2], []])

        assert url == "http://run"
        assert langfuse_cls.call_args.kwargs["base_url"] == "http://localhost:3100"
        client.create_dataset.assert_called_once()
        assert client.create_dataset.call_args.kwargs["name"] == DATASET_NAME
        first = client.create_dataset_item.call_args_list[0].kwargs
        assert first["id"] == f"{DATASET_NAME}-q01"   # item ids are unique across datasets
        assert first["input"] == {"question": "When was Ada born?"}
        assert first["expected_output"] == {"source": "adalovelace", "keyword": "1815"}

        run = client.run_experiment.call_args.kwargs
        assert run["run_name"] == "chunk-80"
        assert run["metadata"] == {"chunk_size": "80"}
        assert [item.id for item in run["data"]] == [f"{DATASET_NAME}-q01", f"{DATASET_NAME}-q02"]
        task = run["task"]
        assert task(item=run["data"][0]) == {"rank": 2, "relevant_ranks": [2]}
        assert task(item=run["data"][1]) == {"rank": None, "relevant_ranks": []}
        client.flush.assert_called_once()

    def test_missing_key_warns_and_publishes_nothing(self, monkeypatch, capsys):
        for name, value in LANGFUSE_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY")
        with patch("langfuse.Langfuse") as langfuse_cls:
            assert log_to_langfuse(None, {}, ITEMS, [[2], []]) is None
        langfuse_cls.assert_not_called()
        assert "LANGFUSE_SECRET_KEY" in capsys.readouterr().err

    def test_langfuse_failure_warns_and_does_not_raise(self, monkeypatch, capsys):
        for name, value in LANGFUSE_ENV.items():
            monkeypatch.setenv(name, value)
        with patch("langfuse.Langfuse") as langfuse_cls:
            langfuse_cls.return_value.create_dataset.side_effect = ConnectionError("refused")
            assert log_to_langfuse(None, {}, ITEMS, [[2], []]) is None
        assert "refused" in capsys.readouterr().err
