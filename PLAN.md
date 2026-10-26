# PLAN — implementation

How mini-rag gets built, step by step, test-first, against the contracts in [TDD.md](TDD.md).

Each step follows the same loop:
1. **Red**: write the tests for that step's contracts and watch them fail.
2. **Green**: write the smallest change that makes them pass.
3. **Refactor**: clean up while the tests stay green.
4. Commit.

Steps 1–4 and 7 run without API keys. Step 5 tests mock OpenAI. Only step 8 calls paid APIs.

---

## Step 0 — Environment and baseline ✅

- [x] `python -m venv .venv && pip install -r requirements-dev.txt`
- [x] `cp .env.example .env`. The key stays empty for now; retrieval and tests don't need it.
- [x] Set up the test layout: `tests/` holds the tests, and `pyproject.toml` sets `pythonpath` and `testpaths`. Shared fixtures go into `tests/conftest.py` in the step that first needs them (Steps 2 and 3).
- [x] Check that `pytest` runs. It collects no tests yet: Steps 1–6 add each module together with its tests.

No application code is committed in this step. The earlier prototype is kept only as local reference; each module is rebuilt test-first in its own step.

**Done when**: the dependencies install cleanly and `pytest` runs. ✅

## Step 1 — Parsing (`parsing.py`) ✅

Tests first:
- [x] `TxtParser.parse_file`: returns one page, `page_number == 1`, `file_type == "txt"`, and the text exactly as it is in the file.
- [x] `ParserFactory.supports`: `.txt` and `.TXT` → `True`; `.md` and files with no extension → `False`.
- [x] `ParserFactory.get_parser`: raises `NotImplementedError` for unsupported files.

Implementation and cleanup:
- [x] Rename `txtParser` to `TxtParser`.
- [x] Remove the fields nobody uses from `Page` (`chunks`, `embeddings`) and the `numpy` import.
- [x] Drop the old `lint_text` cleanup. Measured on the three data files, the token IDs are identical with and without it, because the tokenizer already does the same cleanup.

**Done when**: all parser tests pass with no model loaded. ✅ 11 passed in 0.01 s.

## Step 2 — Chunking (`chunker.py`)

Tests first, using a fake tokeniser so they run in milliseconds:
- [ ] 12 tokens, size 5, overlap 20% → windows `[0..4] [4..8] [8..11]`.
- [ ] Text that fits in one chunk → exactly one chunk.
- [ ] Blank text → `[]`.
- [ ] Invalid size or overlap → `ValueError`.
- [ ] `source == str(path)`, and `chunk_id`s are unique.
- [ ] No chunk lies entirely inside the previous one, and the last chunk ends at the last token.
- [ ] Chunk text is an exact slice of the original text: `"Title\n\nBody"` stays `"Title\n\nBody"`, not `"title body"`.

One test with the real model:
- [ ] Create `tests/conftest.py` with a session-scoped `model` fixture, so the embedding model loads once per test run.
- [ ] Every chunk of `data/*.txt` re-encodes to no more than `max_seq_length` tokens.

Implementation and cleanup:
- [ ] Leave out the old `clean_form_text`, which was never called.
- [ ] Cut chunks from the original text using the tokenizer's character positions (`return_offsets_mapping=True`), instead of decoding tokens.
- [ ] Make the chunk size and overlap constructor arguments with the current defaults, and pass `chunk_size_tokens` through `RAGBot.__init__`. The retrieval eval's `--chunk-size` option depends on this.

**Done when**: the chunker tests pass, and the boundary cases are tested explicitly.

## Step 3 — Indexing (`RAGBot.read_and_embed_data`)

Tests first:
- [ ] Add a session-scoped `bot` fixture to `tests/conftest.py` (built on `model`, indexed over `data/`), and an `empty_bot` fixture that shares the model but has its own empty index.
- [ ] Indexing the sample `data/` gives `len(chunks) == embeddings.shape[0]`, and every row has norm 1.
- [ ] Sources are relative (`"jupiter.txt"`, not `"data/jupiter.txt"`).
- [ ] A temporary folder containing `.DS_Store` and a blank `.txt` → `ValueError`.
- [ ] Add `planet.TXT` to that folder and index it twice → exactly one chunk.
- [ ] A folder that doesn't exist → `FileNotFoundError`.
- [ ] A failed reload leaves the previous index in place (atomic swap).
- [ ] Files are processed in sorted order, so chunk order is stable from run to run.

**Done when**: indexing tests pass offline, and `RAGBot()` needs no OpenAI key.

## Step 4 — Retrieval (`RAGBot.retrieve`)

Tests first:
- [ ] Ada Lovelace's collaborator → top source is `adalovelace.txt`.
- [ ] The four Galilean moons → `jupiter.txt`.
- [ ] What CRISPR stands for → `crispr.txt`.
- [ ] Scores come back sorted from highest to lowest and lie in `[-1, 1]`.
- [ ] `k` larger than the number of chunks → returns every chunk.
- [ ] `k` of `0`, `-1`, `1.5` or `True` → `ValueError("k must be a positive integer.")`.
- [ ] A blank question, or no index loaded → the exact messages in TDD.md.

**Done when**: retrieval tests pass offline, and a quick look at the top-3 results for each sample question shows they make sense.

## Step 5 — Generation (`RAGBot.ask`)

Tests first, with OpenAI mocked:
- [ ] The request uses `MODEL_NAME` and `temperature=0`.
- [ ] The system prompt contains the grounding, "don't know" and citation instructions.
- [ ] The user message contains numbered `[n] Source:` blocks followed by the question.
- [ ] The response is the answer followed by `Retrieved sources:`, with numbers matching the context blocks.
- [ ] A missing or blank `OPENAI_API_KEY` returns the "Set these variables…" message, and the client is never created.
- [ ] `OpenAIError` returns the "request failed" message.
- [ ] An empty or whitespace answer returns the "no answer" message.
- [ ] The client is created once and reused across calls.
- [ ] A blank question returns `"Please enter a question."` without any API call.

Manual check with a real key:
- [ ] A question for each topic, plus one outside the documents (e.g. a restaurant recommendation), which should get "don't know".

**Done when**: the mocked tests pass, and the manual check answers correctly with correct citations.

## Step 6 — CLI (`main.py`)

Tests first:
- [ ] Running from another working directory still finds `data/`.
- [ ] EOF on stdin → exits with status 0 and no traceback.

**Done when**: `python main.py` works from both the repo root and elsewhere.

## Step 7 — Retrieval eval (CI)

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
- [ ] Implement `eval/retrieval.py` by adapting `eval_recall.py` to mini-rag's API (`TokenChunker`, `chunk_id`, `RAGBot(chunk_size_tokens=...)`). Then implement `eval/checks.py`.
- [ ] Generate `eval/baselines/retrieval.json` and commit it.
- [ ] Optional: run `--chunk-size` at 80, 120, 200 and 256 and record the results in the README, to justify the default.

**Done when**: `pytest` runs the retrieval gate in seconds, with no API keys.

## Step 8 — LLM answer eval (on demand)

- [ ] Add `requirements-eval.txt` containing `mlflow` (a version with `mlflow.genai`) and `anthropic`. Add `mlruns/` to `.gitignore`.
- [ ] Implement `eval/llm_eval.py` with the scorers listed in TDD.md. Smoke-test it with `--limit 3`.
- [ ] Do a full run and check about 10 judge verdicts by hand. If the judge disagrees with you, tighten the `refusal` instructions.
- [ ] Record the baseline run name and its scores in the README.

**Done when**: one command produces a run in `mlflow ui` with overall scores and scores per `answer_type`.

## Step 9 — Polish and publish

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
