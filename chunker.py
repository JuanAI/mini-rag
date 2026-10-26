from pathlib import Path
from uuid import uuid4

from parsing import Chunk


class Tokeniser:
    def __init__(self, model) -> None:
        self.tokenizer = model.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required: chunking needs character positions.")
        self.max_tokens = model.max_seq_length - self.tokenizer.num_special_tokens_to_add()

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        # Full documents may exceed the model limit; only their chunks get embedded.
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        return encoding["input_ids"], encoding["offset_mapping"]


class TokenChunker:
    def __init__(
        self,
        tokeniser: Tokeniser,
        chunk_size_tokens: int = 200,
        chunk_overlap_percentage: float = 0.2,
    ) -> None:
        if chunk_size_tokens <= 0 or not 0 <= chunk_overlap_percentage < 1:
            raise ValueError("Chunk size must be positive and overlap must be in [0, 1).")
        self.tokeniser = tokeniser
        self.chunk_size_tokens = min(chunk_size_tokens, tokeniser.max_tokens)
        self.chunk_overlap_percentage = chunk_overlap_percentage

    def chunk(self, text: str, path: Path | str = "") -> list[Chunk]:
        _, spans = self.tokeniser.encode(text)
        overlap_tokens = int(self.chunk_size_tokens * self.chunk_overlap_percentage)

        chunks: list[Chunk] = []
        start = 0
        while start < len(spans):
            end = min(start + self.chunk_size_tokens, len(spans))
            # Slice the original text so case, punctuation and line breaks survive.
            chunks.append(Chunk(
                text=text[spans[start][0]:spans[end - 1][1]],
                chunk_id=uuid4(),
                source=str(path),
            ))
            if end == len(spans):
                break
            start = end - overlap_tokens

        return chunks
