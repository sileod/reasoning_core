"""Compatibility import; use :mod:`reasoning_core.integrations.synlogic`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("reasoning_core.integrations.synlogic")
