from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    text: str
    type: str

class Parser(ABC):
    file_type = None 

    @abstractmethod
    def parse(self, path: Path) -> Document:
        pass

class txtParser(Parser):
    file_type = "txt"

    def parse(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")

        return Document(text=text, type=self.file_type)


class ParserFactory:
    @staticmethod
    def get_parser(path: Path) -> Parser | None:
        file_type = path.suffix.lstrip(".").lower()

        if file_type == txtParser.file_type:
            return txtParser()

        logger.warning(f"File type {file_type} does not have parser")