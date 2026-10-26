import re
from pathlib import Path

import pytest

from chunker import TokenChunker, Tokeniser

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class FakeTokeniser:
    """One token per whitespace-separated word, with real character positions."""

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        return list(range(len(spans))), spans


def words(n: int) -> str:
    return " ".join(f"t{i}" for i in range(n))


class TestTokenChunker:
    def test_windows_overlap_and_stop_at_last_token(self):
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=5, chunk_overlap_percentage=0.2)

        chunks = chunker.chunk(words(12))

        assert [c.text for c in chunks] == [
            "t0 t1 t2 t3 t4", "t4 t5 t6 t7 t8", "t8 t9 t10 t11",
        ]

    def test_text_that_fits_gives_one_chunk(self):
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=5)
        assert [c.text for c in chunker.chunk(words(5))] == [words(5)]

    def test_exact_multiple_has_no_redundant_tail(self):
        # 9 tokens, size 5, overlap 1: [0..4] [4..8]; a third chunk would sit inside the second.
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=5, chunk_overlap_percentage=0.2)
        assert [c.text for c in chunker.chunk(words(9))] == ["t0 t1 t2 t3 t4", "t4 t5 t6 t7 t8"]

    def test_zero_overlap(self):
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=2, chunk_overlap_percentage=0)
        assert [c.text for c in chunker.chunk(words(5))] == ["t0 t1", "t2 t3", "t4"]

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t\n"])
    def test_blank_text_gives_no_chunks(self, blank):
        assert TokenChunker(FakeTokeniser()).chunk(blank) == []

    @pytest.mark.parametrize("size, overlap", [(0, 0.2), (-1, 0.2), (5, -0.1), (5, 1.0), (5, 1.5)])
    def test_invalid_settings_raise(self, size, overlap):
        with pytest.raises(ValueError):
            TokenChunker(FakeTokeniser(), chunk_size_tokens=size, chunk_overlap_percentage=overlap)

    def test_size_is_capped_by_model_limit(self):
        assert TokenChunker(FakeTokeniser(max_tokens=50), chunk_size_tokens=200).chunk_size_tokens == 50

    def test_defaults(self):
        chunker = TokenChunker(FakeTokeniser())
        assert (chunker.chunk_size_tokens, chunker.chunk_overlap_percentage) == (200, 0.2)

    def test_text_is_an_exact_slice_of_the_original(self):
        text = "Title\n\nBody   Text,\twith Capitals.\nEnd"
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=3, chunk_overlap_percentage=0)

        chunks = chunker.chunk(text)

        assert [c.text for c in chunks] == ["Title\n\nBody   Text,", "with Capitals.\nEnd"]
        assert all(c.text in text for c in chunks)

    def test_source_and_unique_ids(self):
        chunker = TokenChunker(FakeTokeniser(), chunk_size_tokens=3)
        chunks = chunker.chunk(words(10), Path("docs/a.txt"))

        assert {c.source for c in chunks} == {str(Path("docs/a.txt"))}
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_source_defaults_to_empty(self):
        assert TokenChunker(FakeTokeniser()).chunk("a b")[0].source == ""


class TestWithRealModel:
    def test_tokeniser_limit_and_positions(self, model):
        tokeniser = Tokeniser(model)
        text = "Ada Lovelace met Babbage."

        ids, spans = tokeniser.encode(text)

        assert tokeniser.max_tokens == model.max_seq_length - model.tokenizer.num_special_tokens_to_add()
        assert len(ids) == len(spans) > 0
        assert text[spans[0][0]:spans[0][1]].lower() == "ada"

    def test_keeps_case_and_line_breaks(self, model):
        chunks = TokenChunker(Tokeniser(model)).chunk("Title\n\nBody Text")
        assert [c.text for c in chunks] == ["Title\n\nBody Text"]

    def test_every_data_chunk_fits_the_model(self, model):
        tokeniser = Tokeniser(model)
        chunker = TokenChunker(tokeniser)
        for path in sorted(DATA_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for chunk in chunker.chunk(text, path.name):
                assert chunk.text in text
                ids, _ = tokeniser.encode(chunk.text)
                assert len(ids) <= tokeniser.max_tokens, (path.name, chunk.text[:60])
