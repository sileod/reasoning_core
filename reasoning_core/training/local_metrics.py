"""Compatibility import; use :mod:`reasoning_core.evaluation.training.local_metrics`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("reasoning_core.evaluation.training.local_metrics")
