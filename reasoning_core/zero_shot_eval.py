"""Compatibility import; use :mod:`reasoning_core.evaluation.zero_shot`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("reasoning_core.evaluation.zero_shot")
