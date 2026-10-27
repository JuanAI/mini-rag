# PLAN — implementation

How mini-rag gets built, step by step, test-first, against the contracts in [TDD.md](TDD.md).

Each step follows the same loop:
1. **Red**: write the tests for that step's contracts and watch them fail.
2. **Green**: write the smallest change that makes them pass.
3. **Refactor**: clean up while the tests stay green.
4. Commit.

Steps 1–5 run without API keys; Step 3 tests mock OpenAI. Only Step 6 calls paid APIs.

---

## Step 0 — Environment and baseline ✅

- [x] `python -m venv .venv && pip install -r requirements-dev.txt`
- [x] `cp .env.example .env`. The key stays empty for now; retrieval and tests don't need it.
- [x] Set up the test layout: `tests/` holds the tests, and `pyproject.toml` sets `pythonpath` and `testpaths`. Shared fixtures go into `tests/conftest.py` in the step that first needs them (Steps 2 and 3).
- [x] Check that `pytest` runs. It collects no tests yet: Steps 1–4 add each module together with its tests.

No application code is committed in this step. The earlier prototype is kept only as local reference; each module is rebuilt test-first in its own step.

**Done when**: the dependencies install cleanly and `pytest` runs. ✅

## Step 1 — Parsing (`parser.py`) ✅

Tests first:
- [x] `TxtParser.parse_file`: returns one page, `page_number == 1`, `file_type == "txt"`, and the text exactly as it is in the file.
- [x] `ParserFactory.supports`: `.txt` and `.TXT` → `True`; `.md` and files with no extension → `False`.
- [x] `ParserFactory.get_parser`: raises `NotImplementedError` for unsupported files.

Implementation and cleanup:
- [x] Rename `txtParser` to `TxtParser`.
- [x] Remove the fields nobody uses from `Page` (`chunks`, `embeddings`) and the `numpy` import.
- [x] Drop the old `lint_text` cleanup. Measured on the three data files, the token IDs are identical with and without it, because the tokenizer already does the same cleanup.

**Done when**: all parser tests pass with no model loaded. ✅ 11 passed in 0.01 s.

**Refactor (after Step 1):** `parsing.py` was replaced with the parser from an earlier version of this project:
- `Document` is just `text` and `type`, and `txtParser.parse()` returns it.
- `ParserFactory.get_parser()` returns `None` with a warning for unsupported files, instead of raising, and there is no `supports()`.
- `Chunk` moves to `chunker.py`.
- `tests/test_parser.py` was rewritten to match: 9 tests.

## Step 2 — Chunking (`chunker.py`) ✅

`chunker.py` is the chunker from the earlier version of this project, copied exactly.

Tests (`tests/test_chunker.py`, 11):
- [x] Defaults: `chunk_size == 80`, `chunk_overlap == 0.2`.
- [x] Each window holds `chunk_size + overlap` tokens and starts `chunk_size` tokens after the previous one. Size 5 → `[0..5] [5..10] [10..11]`; 200 tokens with the defaults → windows of 96, 96 and 40 starting at 0, 80 and 160.
- [x] Consecutive chunks share exactly `overlap` (16) tokens.
- [x] Text that fits gives one chunk; blank text gives `[]`.
- [x] `id` counts from 0 within each document and restarts for the next one; `source == str(path)`.
- [x] With the real model:
  - `Tokeniser` adds no special tokens;
  - decoding lowercases the text;
  - every chunk of `data/*.txt` fits the model's limit (at most 99 of 254 tokens).
- [x] `tests/conftest.py` provides a session-scoped `model` fixture.

Known behaviour, kept as is:
- Chunk text is **decoded** from tokens, so it comes out lowercase and without line breaks.
- `id`s restart at 0 for each file, so they aren't unique across the index. `rag.py` cites `[chunk.id]` plus the source, so a citation needs both.
- Size and overlap are attributes (`chunker.chunk_size = N`), not constructor arguments.

**Done when**: the chunker tests pass. ✅ 20 passed in total (9 parser + 11 chunker) in ~4 s. The data files give 111, 114 and 147 chunks.

## Step 3 — RAG engine (`rag.py`) ✅

Brings in `rag.py` from the earlier version of this project. Indexing, retrieval and answer generation are covered in this one step.

Changes from the earlier version:
- [x] `parsing.py` renamed back to `parser.py`, the name `rag.py` imports.
- [x] The embedding model is bundled in `models/all-MiniLM-L6-v2` (87 MB), so the relative path in `rag.py` works on a fresh clone. `models/` is no longer ignored by git.
- [x] Both providers are supported: Azure OpenAI when `AZURE_OPENAI_ENDPOINT` is set, otherwise OpenAI. The model comes from `MODEL_NAME`, default `gpt-4.1-mini`.
- [x] `httpx.Client(verify=False)` removed, so HTTPS certificates are checked again.
- [x] The OpenAI client is still created in `__init__` (kept as is), so a key must be set even to index.
- [x] Unused `httpx` and `pandas` imports removed, so neither is needed in `requirements.txt`.
- [x] Prompt typos fixed: missing spaces between sentences, "dont't" → "don't", "Question" → "Question: ", and a full stop after "Be concise".

Tests (`tests/test_rag.py`, 15, with OpenAI mocked):
- [x] **Client choice:**
  - OpenAI without an Azure endpoint, Azure with one, each with the right settings;
  - the bundled model path is used.
- [x] **Indexing:**
  - one unit-length vector per chunk;
  - all three documents are indexed;
  - `.DS_Store` and `.md` are skipped.
- [x] **Retrieval:**
  - the Ada Lovelace, Jupiter and CRISPR questions put a chunk from the right document first;
  - `k` chunks are sent;
  - with fewer chunks than `k`, all of them are sent.
- [x] **Generation:**
  - model name and `temperature=0`;
  - the exact corrected system prompt;
  - the user message format;
  - `MODEL_NAME` defaults to `gpt-4.1-mini`;
  - the reply lists the sources;
  - `OpenAIError` is re-raised with context.
- [x] `tests/conftest.py` loads the bundled model and provides a session-scoped `bot` with a dummy key.

Manual check with a real key:
- [ ] A question for each topic, plus one outside the documents (e.g. a restaurant recommendation), which should get "don't know".

**Done when**: the mocked tests pass offline (✅ 35 passed in ~5 s: 9 parser + 11 chunker + 15 rag), and the manual check answers correctly with correct citations.

## Step 4 — CLI (`main.py`) ✅

`main.py` comes from the earlier version of this project, with two additions:
- Ctrl-D or Ctrl-C ends the session cleanly instead of printing a traceback.
- A failed request (e.g. an expired API key) is reported and the session continues.

 It runs from the repo root, because `data/` and `models/` are relative paths.

Tests first (`tests/test_cli.py`, 5):
- [x] With a fake bot, each question gets `RAGBot: {answer}` and the greeting is printed.
- [x] Ctrl-D (`EOFError`) and Ctrl-C (`KeyboardInterrupt`) end the loop without an exception. Both failed against the exact copy.
- [x] A real run from the repo root with empty stdin exits with status 0 and no traceback.
- [x] When `ask()` raises, `RAGBot: Sorry, that request failed: …` is printed and the next question is still answered. Found in a real run with an expired key.

VS Code setup (`.vscode/`, from the earlier version of this project):
- [x] `launch.json`: a "RAGBot: main.py" run configuration that uses `.venv`, the integrated terminal (for the input loop) and the repo root as working directory.
- [x] `settings.json`: `.venv` as the interpreter, and the Test Explorer running pytest on `tests/`.
- [x] Comments give the real setup commands (`python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`), not `make venv`.

**Done when**: `python main.py` works from the repo root, and the VS Code run button starts the bot. ✅ 40 passed in ~7 s (9 parser + 11 chunker + 15 rag + 5 cli).

## Step 5 — Retrieval eval (CI)

- [ ] Add `requirements-dev.txt` containing `pytest`. No retrieval-eval library is needed.
- [ ] Port the 50 questions from the earlier `eval_recall.py` into `eval/dataset.jsonl` (`id`, `question`, `answer_type`, `keyword`, `source`, `expected_answer`). Add about 5 unanswerable questions.

Tests first:
- [ ] `norm("£3,000")`, `norm("MIL-STD-1815")` → `"3 000"`, `"mil std 1815"`.
- [ ] `invalid_keywords(bot, dataset) == []`.
- [ ] `rank_of_answer`:
  - with a fake index → the expected rank;
  - when the keyword exists only in another document → `None`.
- [ ] `evaluate_retrieval`:
  - every metric is in `[0, 1]`, and `recall@k` never decreases as k grows;
  - a second run gives the same numbers.
- [ ] `invalid_citations("A [1] B [4]", 3) == [4]`, plus `[0]`, no citations, and repeated citations.
- [ ] `recall@3` and `mrr` are each ≥ baseline − 0.02.

Then:
- [ ] Implement `eval/retrieval.py` by adapting `eval_recall.py` to mini-rag's API (`Chunker`, `chunk.id`, `bot.chunker.chunk_size`). Then implement `eval/checks.py`.
- [ ] Generate `eval/baselines/retrieval.json` and commit it.
- [ ] Optional: run `--chunk-size` at 80, 120, 200 and 256 and record the results in the README, to justify the default.

**Done when**: `pytest` runs the retrieval gate in seconds, with no API keys.

## Step 6 — LLM answer eval (on demand)

- [ ] Add `requirements-eval.txt` containing `mlflow` (a version with `mlflow.genai`) and `anthropic`. Add `mlruns/` to `.gitignore`.
- [ ] Implement `eval/llm_eval.py` with the scorers listed in TDD.md. Smoke-test it with `--limit 3`.
- [ ] Do a full run and check about 10 judge verdicts by hand. If the judge disagrees with you, tighten the `refusal` instructions.
- [ ] Record the baseline run name and its scores in the README.

**Done when**: one command produces a run in `mlflow ui` with overall scores and scores per `answer_type`.

## Step 7 — Polish and publish

- [ ] Search the repo for secrets: no `sk-` strings, `.env` is ignored by git, and `.env.example` contains only placeholders.
- [ ] Check that README examples match real output. Replace the illustrative sample session with a real one.
- [ ] Add a GitHub Actions workflow (`ci.yml`) that, on every push and PR, runs `pytest` on Python 3.10–3.12, covering the unit tests, the retrieval gate and the citation checks. Cache the Hugging Face model between runs. It needs no secrets.
- [ ] Optional: a `llm-eval.yml` workflow triggered manually (`workflow_dispatch`), using API keys stored as repo secrets.
- [ ] Optional: a `ruff` config.
- [ ] `git init`, make the first commit, add the remote `git@github.com:JuanAI/mini-rag.git`, and push to `main`.

**Done when**: CI is green on `main`, and the repo on GitHub matches the README.

---

## Out of scope for now

Persistence for the vector index, a reranker, more file types (PDF, Markdown), conversation memory and streaming responses. Each would be its own follow-up, with contracts added to TDD.md first.
