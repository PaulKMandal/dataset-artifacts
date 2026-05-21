#!/usr/bin/env python3
"""Build per-example clean-to-adversarial failure features.

This script aligns raw prediction JSONL files by SQuAD example id. It does not
pretend that train-set cartography scores directly exist for dev/adversarial
items; cartography columns are left blank unless a provided cartography file has
matching example IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from argparse import Namespace
from pathlib import Path

from qa_metrics import normalize_answer
