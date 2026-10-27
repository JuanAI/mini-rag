from pathlib import Path
from types import SimpleNamespace

from chunker import Chunk, Chunker, Tokeniser

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class FakeTokenizer:
    """One token per whitespace-separated word; decoding joins the words back with spaces."""

    def __init__(self) -> None:
        self.vocab: list[str] = []

    def encode(self, text, add_special_tokens=True):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab.append(word)
            ids.append(self.vocab.index(word))
        return ids

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(self.vocab[t] for t in tokens)


def fake_model():
    return SimpleNamespace(tokenizer=FakeTokenizer())


def words(n: int) -> str:
    return " ".join(f"t{i}" for i in range(n))


class TestChunker:
    def test_defaults(self):
        chunker = Chunker(fake_model())
        assert (chunker.chunk_size, chunker.chunk_overlap) == (80, 0.2)

    def test_windows_are_size_plus_overlap_and_step_by_size(self):
        # size 5, overlap int(0.2 * 5) = 1: each window holds 6 tokens and starts 5 after the last.
        chunker = Chunker(fake_model())
        chunker.chunk_size = 5

        chunks = chunker.get_chunks(words(12), Path("a.txt"))

        assert [c.text for c in chunks] == [
            "t0 t1 t2 t3 t4 t5", "t5 t6 t7 t8 t9 t10", "t10 t11",
        ]

    def test_default_windows(self):
        # 200 tokens with the defaults: windows of 96 tokens starting at 0, 80 and 160.
        chunks = Chunker(fake_model()).get_chunks(words(200), Path("a.txt"))

        assert [len(c.text.split()) for c in chunks] == [96, 96, 40]
        assert [c.text.split()[0] for c in chunks] == ["t0", "t80", "t160"]

    def test_consecutive_chunks_share_the_overlap(self):
        chunks = Chunker(fake_model()).get_chunks(words(200), Path("a.txt"))
        for previous, current in zip(chunks, chunks[1:]):
            assert previous.text.split()[-16:] == current.text.split()[:16]

    def test_text_that_fits_gives_one_chunk(self):
        chunks = Chunker(fake_model()).get_chunks(words(50), Path("a.txt"))
        assert [c.text for c in chunks] == [words(50)]

    def test_blank_text_gives_no_chunks(self):
        assert Chunker(fake_model()).get_chunks("  \n ", Path("a.txt")) == []

    def test_ids_count_from_zero_and_source_is_the_path(self):
        chunker = Chunker(fake_model())
        chunker.chunk_size = 5

        chunks = chunker.get_chunks(words(12), Path("docs/a.txt"))

        assert [c.id for c in chunks] == [0, 1, 2]
        assert {c.source for c in chunks} == {str(Path("docs/a.txt"))}

    def test_ids_restart_for_each_document(self):
        chunker = Chunker(fake_model())
        assert chunker.get_chunks("a b", Path("a.txt"))[0].id == 0
        assert chunker.get_chunks("c d", Path("b.txt"))[0].id == 0

    def test_chunk_fields(self):
        assert Chunk(id=0, text="t", source="a.txt") == Chunk(0, "t", "a.txt")


class TestWithRealModel:
    def test_tokeniser_round_trip(self, model):
        tokeniser = Tokeniser(model)
        ids = tokeniser.encode("Ada Lovelace met Babbage.")
        assert model.tokenizer.cls_token_id not in ids
        assert tokeniser.decode(ids) == "ada lovelace met babbage."

    def test_every_data_chunk_fits_the_model(self, model):
        chunker = Chunker(model)
        limit = model.max_seq_length - model.tokenizer.num_special_tokens_to_add()
        for path in sorted(DATA_DIR.glob("*.txt")):
            chunks = chunker.get_chunks(path.read_text(encoding="utf-8"), Path(path.name))
            assert chunks
            for chunk in chunks:
                assert len(model.tokenizer.encode(chunk.text, add_special_tokens=False)) <= limit
