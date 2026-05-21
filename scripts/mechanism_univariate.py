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

def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    precisions = []
    tp = 0
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else float("nan")
