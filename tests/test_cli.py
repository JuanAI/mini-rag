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

    def read_and_embed_data(self, folder_path):
        self.folder = folder_path

    def ask(self, question):
        return f"answer to {question}"


def run_main(inputs):
    """Run main.py in-process with a fake bot; `inputs` are returned by input() or raised."""
    def fake_input(prompt=""):
        item = inputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    with patch("rag.RAGBot", FakeBot), patch("builtins.input", fake_input):
        runpy.run_path(MAIN, run_name="__main__")


def test_answers_each_question(capsys):
    run_main(["What is CRISPR?", "Who was Ada?", EOFError()])

    out = capsys.readouterr().out
    assert "Hi, I'm your RAGBot. Ask me anything about Jupiter, Ada Lovelace, or CRISPR" in out
    assert "RAGBot: answer to What is CRISPR?\n" in out
    assert "RAGBot: answer to Who was Ada?\n" in out


@pytest.mark.parametrize("stop", [EOFError(), KeyboardInterrupt()])
def test_ctrl_d_or_ctrl_c_exits_cleanly(stop):
    run_main([stop])  # must return normally, not raise


def test_real_run_from_repo_root_exits_cleanly_on_eof():
    env = {**os.environ, "OPENAI_API_KEY": "test-key", "TOKENIZERS_PARALLELISM": "false"}
    env.pop("AZURE_OPENAI_ENDPOINT", None)

    result = subprocess.run(
        [sys.executable, "main.py"], cwd=ROOT, env=env,
        input="", capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Hi, I'm your RAGBot" in result.stdout
