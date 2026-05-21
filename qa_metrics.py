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
