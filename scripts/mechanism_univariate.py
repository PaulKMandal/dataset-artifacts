#!/usr/bin/env python3
"""Compute univariate ROC-AUC/AP for mechanism features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_FEATURES = [
    "confidence",
    "variability",
    "correctness",
    "context_length",
    "answer_length",
    "answer_position_normalized",
    "question_answer_sentence_overlap",
    "question_distractor_overlap",
    "num_added_sentences",
]

def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0.0
    for p in pos:
        greater += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(greater / (len(pos) * len(neg)))
