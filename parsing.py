from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass
class Chunk:
    text: str
    chunk_id: UUID | str
    source: str = ""


@dataclass
class Page:
    page_number: int
    text: str


@dataclass
class Document:
    pages: list[Page]
    file_type: str


class Parser(ABC):
    file_type: str

    @abstractmethod
    def parse_file(self, file_path: Path) -> Document:
        pass


class TxtParser(Parser):
    file_type = "txt"

    def parse_file(self, file_path: Path) -> Document:
        text = file_path.read_text(encoding="utf-8")
        return Document(pages=[Page(page_number=1, text=text)], file_type=self.file_type)


class ParserFactory:
    _parsers: dict[str, type[Parser]] = {TxtParser.file_type: TxtParser}

    @staticmethod
    def _file_type(file_path: Path) -> str:
        return file_path.suffix.lstrip(".").lower()

    def supports(self, file_path: Path) -> bool:
        return self._file_type(file_path) in self._parsers

    def get_parser(self, file_path: Path) -> Parser:
        file_type = self._file_type(file_path)
        if file_type not in self._parsers:
            raise NotImplementedError(f"No parser implemented for type {file_type!r}")
        return self._parsers[file_type]()
