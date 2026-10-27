from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from pathlib import Path

@dataclass
class Chunk:
    id: int 
    text: str
    source: str

class Tokeniser:
    def __init__(self, model: SentenceTransformer) -> None:
        self.tokeniser: BertTokenizer  = model.tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokeniser.encode(text,add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokeniser.decode(tokens, skip_special_tokens=True)


class Chunker:
    def __init__(self, model: SentenceTransformer) -> None:
        self.tokeniser = Tokeniser(model)
        self.chunk_size = 80
        self.chunk_overlap = 0.2

    def get_chunks(self, text: str, path: Path) -> list[Chunk]:
        tokens = self.tokeniser.encode(text)
        overlap_tokens = int(self.chunk_overlap * self.chunk_size)

        start = 0
        idx = 0
        chunks = []

        while start < len(tokens):
            end = start + min(self.chunk_size + overlap_tokens, len(tokens))
            chunks.append(Chunk(id=idx,text=self.tokeniser.decode(tokens[start:end]), source=str(path)))

            if end == len(tokens):
                break 

            start = end - overlap_tokens
            idx += 1

        return chunks
