"""Pytest must set this before any test module can import transformers."""

import os

# transformers imports tensorflow eagerly (image_transforms.py) whenever it can find it. This
# project is torch-only, so switch the probe off: it drops a multi-second import from every arm,
# and TF's x86 build aborts outright on hosts without SSE4.1 (the G5K frontends).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
