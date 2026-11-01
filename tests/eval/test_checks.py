"""eval/checks.py: the deterministic citation check."""
import pytest

from eval.checks import invalid_citations


class TestInvalidCitations:
    @pytest.mark.parametrize("answer, allowed, expected", [
        ("Io [3] and Europa [7].", {3, 7}, []),
        ("A [3] B [4]", {3, 7}, [4]),
        ("A [0]", {3}, [0]),
        ("No citations here.", {3}, []),
        ("A [4] B [4] C [9]", {3}, [4, 9]),
        ("Range [2-3] and text [x] are not citations.", {1}, []),
    ])
    def test_cases(self, answer, allowed, expected):
        assert invalid_citations(answer, allowed) == expected
