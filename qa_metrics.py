"""SQuAD-style exact-match and token-F1 helpers.

These helpers are intentionally small and dependency-free so they can be used
both by ``run.py`` when saving raw predictions and by analysis scripts that need
per-example correctness labels.
"""

from __future__ import annotations

import collections
import re
import string
from typing import Iterable

def normalize_answer(text: str) -> str:
    """Lowercase and remove punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
