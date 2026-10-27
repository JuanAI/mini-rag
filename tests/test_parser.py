import logging
from pathlib import Path

import pytest

from parser import Document, ParserFactory, txtParser


class TestTxtParser:
    def test_returns_text_unchanged(self, tmp_path):
        content = "Title\n\n\nBody  text, with Capitals.\n"
        path = tmp_path / "doc.txt"
        path.write_text(content, encoding="utf-8")

        assert txtParser().parse(path) == Document(text=content, type="txt")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            txtParser().parse(tmp_path / "missing.txt")

    def test_non_utf8_raises(self, tmp_path):
        path = tmp_path / "latin1.txt"
        path.write_bytes("café".encode("latin-1"))
        with pytest.raises(UnicodeDecodeError):
            txtParser().parse(path)


class TestParserFactory:
    @pytest.mark.parametrize("name", ["notes.txt", "NOTES.TXT"])
    def test_txt_gets_txt_parser(self, name):
        parser = ParserFactory.get_parser(Path(name))
        assert isinstance(parser, txtParser)
        assert parser.file_type == "txt"

    @pytest.mark.parametrize("name", ["notes.md", "README", ".DS_Store"])
    def test_unsupported_returns_none_and_warns(self, name, caplog):
        with caplog.at_level(logging.WARNING, logger="parser"):
            assert ParserFactory.get_parser(Path(name)) is None
        assert "does not have parser" in caplog.text

    def test_decides_from_name_without_reading(self):
        # The file doesn't exist: the factory must decide from the name alone.
        assert isinstance(ParserFactory.get_parser(Path("/nonexistent/a.txt")), txtParser)
