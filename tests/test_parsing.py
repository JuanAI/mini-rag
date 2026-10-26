from pathlib import Path

import pytest

from parsing import Chunk, Document, Page, ParserFactory, TxtParser


class TestTxtParser:
    def test_single_page_with_text_unchanged(self, tmp_path):
        # Cleanup is left to the tokenizer; the parser must not alter the text.
        content = "Title\n\n\nBody  text, with Capitals.\n"
        path = tmp_path / "doc.txt"
        path.write_text(content, encoding="utf-8")

        document = TxtParser().parse_file(path)

        assert document == Document(pages=[Page(page_number=1, text=content)], file_type="txt")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TxtParser().parse_file(tmp_path / "missing.txt")

    def test_non_utf8_raises(self, tmp_path):
        path = tmp_path / "latin1.txt"
        path.write_bytes("café".encode("latin-1"))
        with pytest.raises(UnicodeDecodeError):
            TxtParser().parse_file(path)


class TestParserFactory:
    @pytest.mark.parametrize("name, supported", [
        ("notes.txt", True),
        ("NOTES.TXT", True),
        ("notes.md", False),
        ("README", False),
        (".DS_Store", False),
    ])
    def test_supports_by_extension_without_reading(self, name, supported):
        # The files don't exist: supports() must decide from the name alone.
        assert ParserFactory().supports(Path("/nonexistent") / name) is supported

    def test_get_parser_matches_extension(self):
        parser = ParserFactory().get_parser(Path("NOTES.TXT"))
        assert isinstance(parser, TxtParser)
        assert parser.file_type == "txt"

    def test_get_parser_rejects_unsupported(self):
        with pytest.raises(NotImplementedError):
            ParserFactory().get_parser(Path("notes.md"))


def test_chunk_source_defaults_to_empty():
    assert Chunk(text="t", chunk_id="c1").source == ""
