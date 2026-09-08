"""Compatibility import; use :mod:`reasoning_core.integrations.reasoning_gym`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("reasoning_core.integrations.reasoning_gym")
