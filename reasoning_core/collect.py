"""Compatibility import; use :mod:`reasoning_core.generation.collect`."""

import importlib
import sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("reasoning_core.generation.collect", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("reasoning_core.generation.collect")
