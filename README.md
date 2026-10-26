# mini-rag

A small retrieval-augmented generation (RAG) bot built from scratch: no LangChain, no vector database. It reads a folder of `.txt` files, chunks and embeds them locally, retrieves the most relevant passages for a question, and asks an OpenAI model for an answer grounded in, and citing, those passages.

```
Me: What are the four Galilean moons?
RAGBot: Io, Europa, Ganymede and Callisto [1].

Retrieved sources:
[1] jupiter.txt
[2] jupiter.txt
[3] jupiter.txt
```

## How it works

1. **Parse**: `parser.py` reads each `.txt` file and normalizes whitespace and non-printable characters. Other file types are skipped. `ParserFactory` is the place to add new formats.
2. **Chunk**: `chunker.py` splits text by *tokens*, not characters, using the embedding model's own tokenizer. Chunks are up to 200 tokens, capped by the model's input limit, with 20% overlap so context at the boundaries isn't lost. Each chunk keeps its source filename.
3. **Embed**: chunks are batch-encoded with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and L2-normalized, so a dot product equals cosine similarity.
4. **Retrieve**: the question is embedded the same way, scored against every chunk with a single matrix product, and the top *k* (default 3) are returned.
5. **Generate**: the retrieved chunks go into the prompt with numbered source labels. The system prompt tells the model to answer only from that context, treat it as data rather than instructions, cite passages as `[1]`, `[2]`, and say when the answer isn't there.

A NumPy matrix is enough at this scale. There's no persistence, reranker or conversation memory, by design.

## Project structure

```
mini-rag/
├── data/            # sample corpus: Ada Lovelace, Jupiter, CRISPR (from Wikipedia)
├── parser.py        # file → Document/Page, text cleanup
├── chunker.py       # token-based overlapping chunker
├── rag.py           # RAGBot: index, retrieve, ask
├── main.py          # interactive CLI
├── eval/            # eval dataset, retrieval metrics, LLM judge eval
└── test_rag.py      # tests (real embedding model, mocked OpenAI)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI API key
```

`.env` variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (none) | Required to generate answers |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any SentenceTransformer model name or local path |

The embedding model is downloaded from the Hugging Face Hub on first run and cached after that.

## Usage

```bash
python main.py
```

Ask about anything in `data/`. To use your own documents, drop `.txt` files into `data/`.

You can also inspect retrieval without calling the LLM:

```python
from rag import RAGBot

bot = RAGBot()
bot.read_and_embed_data("data")
for chunk, score in bot.retrieve("What does CRISPR stand for?"):
    print(chunk.source, round(score, 3), chunk.text[:80])
```

## Tests

```bash
python -m unittest -v test_rag
```

The tests use the real embedding model and a mocked OpenAI client, so they make no paid API calls.

## Evaluation

| What | Tool | Where it runs |
|---|---|---|
| Retrieval quality: Recall@k and MRR over 50 questions, each with a keyword that marks its answer passage | plain Python (`eval/retrieval.py`) | CI, on every push. Fails if a metric drops below `eval/baselines/retrieval.json` |
| Citation validity | plain Python | CI, on every push |
| Answer groundedness, relevance, correctness, refusal on out-of-scope questions | [MLflow GenAI](https://mlflow.org/docs/latest/genai/) LLM judges | On demand: `python -m eval.llm_eval`, then `mlflow ui` |

CI needs no API keys. The LLM eval calls paid APIs and its scores vary from run to run, so it is run by hand whenever the prompt or model changes.

## Limitations

- Top-k retrieval always returns *something*. Similarity scores are not calibrated confidence.
- Grounding instructions reduce unsupported answers but don't guarantee correctness.
- The whole index lives in memory and is rebuilt on every start.

## Data

The sample texts in `data/` are excerpts from Wikipedia articles, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
