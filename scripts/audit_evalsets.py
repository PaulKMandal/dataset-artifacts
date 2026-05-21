#!/usr/bin/env python3
"""Write a lightweight evalset/table-label audit for the result tree."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_COUNTS = {
    "squad_dev": 10570,
    "addsent": 3560,
    "addonesent": 1787,
}
