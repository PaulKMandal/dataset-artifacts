#!/usr/bin/env python3
"""Create persistent QA subset JSONL files from cartography scores.

Selection rules are intentionally simple and auditable:

* easy: highest confidence, then lowest variability, then highest correctness
* hard: lowest confidence, then lowest correctness, then highest variability
* ambiguous: highest variability, then middle confidence preference
* random: uniform sample without replacement, one persistent file per draw

The script writes a manifest plus a subset_assignments.csv that records exactly
which original SQuAD train indices were selected for every subset/fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from argparse import Namespace
from pathlib import Path
from typing import Iterable


CARTO_SUBSETS = ("easy", "ambiguous", "hard")
