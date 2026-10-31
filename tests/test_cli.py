import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = str(ROOT / "main.py")


class FakeBot:
    def __init__(self):
        self.folder = None
        self.calls = []

    def read_and_embed_data(self, folder_path):
        self.folder = folder_path

    def ask(self, question, session_id=None, user_id=None):
        self.calls.append({"session_id": session_id, "user_id": user_id})
        if question == "fail":
            raise Exception("Error raised from OpenAI model Error code: 401 - key expired")
        return f"answer to {question}"


def run_main(inputs):
    """Run main.py in-process with a fake bot; `inputs` are returned by input() or raised.
    Returns the bot, so tests can check what ask() was called with."""
    bots = []

    def make_bot():
        bots.append(FakeBot())
        return bots[0]

    def fake_input(prompt=""):
        item = inputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    with patch("rag.RAGBot", make_bot), patch("builtins.input", fake_input):
        runpy.run_path(MAIN, run_name="__main__")
    return bots[0]


def test_answers_each_question(capsys):
    run_main(["What is CRISPR?", "Who was Ada?", EOFError()])

    out = capsys.readouterr().out
    assert "Hi, I'm your RAGBot. Ask me anything about Jupiter, Ada Lovelace, or CRISPR" in out
    assert "RAGBot: answer to What is CRISPR?\n" in out
    assert "RAGBot: answer to Who was Ada?\n" in out


def test_failed_request_is_reported_and_the_session_continues(capsys):
    run_main(["fail", "Who was Ada?", EOFError()])

    out = capsys.readouterr().out
    assert "RAGBot: Sorry, that request failed: Error raised from OpenAI model Error code: 401 - key expired\n" in out
    assert "RAGBot: answer to Who was Ada?\n" in out


@pytest.mark.parametrize("stop", [EOFError(), KeyboardInterrupt()])
def test_ctrl_d_or_ctrl_c_exits_cleanly(stop):
    run_main([stop])  # must return normally, not raise


def test_one_session_id_per_run_and_user_from_env(monkeypatch):
    monkeypatch.setenv("RAGBOT_USER_ID", "juan")
    first = run_main(["a", "b", EOFError()])
    second = run_main(["c", EOFError()])

    ids = {call["session_id"] for call in first.calls}
    assert len(ids) == 1 and None not in ids
    assert second.calls[0]["session_id"] not in ids
    assert all(call["user_id"] == "juan" for call in first.calls)


def test_user_defaults_to_the_os_user(monkeypatch):
    monkeypatch.delenv("RAGBOT_USER_ID", raising=False)
    with patch("getpass.getuser", return_value="os-user"):
        bot = run_main(["a", EOFError()])
    assert bot.calls[0]["user_id"] == "os-user"


@pytest.mark.parametrize("stop", [EOFError(), KeyboardInterrupt()])
def test_traces_are_flushed_on_exit(stop):
    with patch("telemetry.flush_telemetry") as flush:
        run_main(["a", stop])
    flush.assert_called_once()


def test_real_run_from_repo_root_exits_cleanly_on_eof():
    env = {**os.environ, "OPENAI_API_KEY": "test-key", "TOKENIZERS_PARALLELISM": "false",
           "LANGFUSE_ENABLED": "false"}
    env.pop("AZURE_OPENAI_ENDPOINT", None)

    result = subprocess.run(
        [sys.executable, "main.py"], cwd=ROOT, env=env,
        input="", capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Hi, I'm your RAGBot" in result.stdout
