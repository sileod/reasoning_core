"""Compatibility import; use :mod:`reasoning_core.evaluation.training.source_signals`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("reasoning_core.evaluation.training.source_signals")
