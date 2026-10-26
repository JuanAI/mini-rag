from pathlib import Path

import pytest

from parsing import Chunk, Document, Page, ParserFactory, TxtParser, lint_text


class TestLintText:
    @pytest.mark.parametrize("raw, expected", [
        ("a\r\nb\rc", "a\nb\nc"),
        ("a \t  b\t\tc", "a b c"),
        ("a\n\n\n\nb", "a\n\nb"),
        ("a\n \t\n  \nb", "a\n\nb"),
        ("a\nb", "a\nb"),
        ("a\x00b\x07c\x1b", "abc"),
        ("a\x00 \x00 b", "a b"),
        ("  \n padded \n  ", "padded"),
    ])
    def test_cleanup_rules(self, raw, expected):
        assert lint_text(raw) == expected

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t\r\n", "\x00\x07"])
    def test_blank_input_returns_empty_string(self, blank):
        assert lint_text(blank) == ""

    @pytest.mark.parametrize("raw", [
        "a\r\n\r\n\r\nb  \t c\x00",
        " x \n\n \n y \n",
        "one\n\ntwo\n\n\n\nthree",
    ])
    def test_idempotent(self, raw):
        once = lint_text(raw)
        assert lint_text(once) == once


class TestTxtParser:
    def test_single_linted_page(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("Title\r\n\r\n\r\nBody  text\x00", encoding="utf-8")

        document = TxtParser().parse_file(path)

        assert document == Document(pages=[Page(page_number=1, text="Title\n\nBody text")], file_type="txt")

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
