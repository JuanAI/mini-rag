# PLAN — implementation

How mini-rag gets built, step by step, test-first, against the contracts in [TDD.md](TDD.md).

Each step follows the same loop:
1. **Red**: write the tests for that step's contracts and watch them fail.
2. **Green**: write the smallest change that makes them pass.
3. **Refactor**: clean up while the tests stay green.
4. Commit.

Steps 1–4 run offline. Only step 5 needs an OpenAI key, and its tests mock OpenAI anyway.

---

## Step 0 — Environment and baseline

- [ ] `python -m venv .venv && pip install -r requirements.txt`
- [ ] `cp .env.example .env` (the key can stay empty for now)
- [ ] Run `python -m unittest -v test_rag` and write down what passes and fails as the starting point.
- [ ] Split the tests into `tests/test_parser.py`, `tests/test_chunker.py`, `tests/test_rag.py` and `tests/test_cli.py`, with a shared fixture that loads the embedding model once per session.

**Done when**: the environment installs cleanly and the test suite runs, even if some tests fail.

## Step 1 — Parsing (`parser.py`)

Tests first:
- [ ] `lint_text`: CRLF/CR → LF, tabs and runs of spaces collapse, three or more newlines become one blank line, non-printable characters are removed, the output is stripped, and running it twice changes nothing.
- [ ] `TxtParser.parse_file`: returns one page, `page_number == 1`, `file_type == "txt"`, and the text is the linted content.
- [ ] `ParserFactory.supports`: `.txt` and `.TXT` → `True`; `.md` and files with no extension → `False`.
- [ ] `ParserFactory.get_parser`: raises `NotImplementedError` for unsupported files.

Implementation and cleanup:
- [ ] Rename `txtParser` to `TxtParser`.
- [ ] Remove the fields nobody uses from `Page` (`chunks`, `embeddings`) and the `numpy` import.

**Done when**: all parser tests pass with no model loaded.

## Step 2 — Chunking (`chunker.py`)

Tests first, using a fake tokeniser so they run in milliseconds:
- [ ] 12 tokens, size 5, overlap 20% → windows `[0..4] [4..8] [8..11]`.
- [ ] Text that fits in one chunk → exactly one chunk.
- [ ] Blank text → `[]`.
- [ ] Invalid size or overlap → `ValueError`.
- [ ] `source == str(path)`, and `chunk_id`s are unique.
- [ ] No chunk lies entirely inside the previous one, and the last chunk ends at the last token.

One test with the real model:
- [ ] Every chunk of `data/*.txt` re-encodes to no more than `max_seq_length` tokens.

Implementation and cleanup:
- [ ] Remove `clean_form_text`, which is never called, or fold anything useful from it into `lint_text`.
- [ ] Make the chunk size and overlap constructor arguments with the current defaults, so tests don't have to modify attributes.

**Done when**: the chunker tests pass, and the boundary cases are tested explicitly.

## Step 3 — Indexing (`RAGBot.read_and_embed_data`)

Tests first:
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

## Step 7 — Polish and publish

- [ ] Search the repo for secrets: no `sk-` strings, `.env` is ignored by git, and `.env.example` contains only placeholders.
- [ ] Check that README examples match real output. Replace the illustrative sample session with a real one.
- [ ] Optional: add a `ruff` config and a GitHub Actions workflow that runs the tests on Python 3.10–3.12, with the Hugging Face model cache saved between runs.
- [ ] `git init`, make the first commit, add the remote `git@github.com:JuanAI/mini-rag.git`, and push to `main`.

**Done when**: CI is green, or the tests pass locally if CI is skipped, and the repo on GitHub matches the README.

---

## Out of scope for now

Persistence for the vector index, a reranker, more file types (PDF, Markdown), conversation memory and streaming responses. Each would be its own follow-up, with contracts added to TDD.md first.
