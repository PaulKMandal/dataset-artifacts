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
