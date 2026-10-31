"""Deterministic answer checks: no LLM needed, so they run in pytest and as eval scorers."""
import re
from collections.abc import Iterable

CITATION = re.compile(r"\[(\d+)\]")


def invalid_citations(answer: str, allowed: Iterable[int]) -> list[int]:
    """Cited numbers ([n]) that aren't in `allowed`, in order of first appearance.

    rag.py labels each retrieved passage with its chunk id, so `allowed` is the
    set of ids that were actually sent to the model.
    """
    allowed = set(allowed)
    invalid: list[int] = []
    for match in CITATION.finditer(answer):
        number = int(match.group(1))
        if number not in allowed and number not in invalid:
            invalid.append(number)
    return invalid
