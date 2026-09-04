"""Low-level execution, provenance, tokenization, and cache support."""

import ast
import base64
import copy
import ctypes
import functools
import hashlib
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import warnings
from io import BytesIO

from appdirs import user_cache_dir


def _tiktoken():
    import tiktoken
    return tiktoken


def _psutil():
    import psutil
    return psutil


@functools.lru_cache(maxsize=None)
def generator_version(package_name):
    module = sys.modules.get(package_name)
    version = getattr(module, "__version__", None)
    if version:
        return version
    try:
        from importlib.metadata import version as package_version
        return package_version(package_name)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def generator_commit():
    for name in ("SOURCE_COMMIT", "GIT_COMMIT", "COMMIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        ).strip()
    except Exception:
        return None


def _strip_docstrings(node):
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            del body[0]
    return node


def _stable_dump(node):
    """Return version-stable AST text, ignoring comments and docstrings."""
    if isinstance(node, ast.AST):
        parts = []
        for field in node._fields:
            value = getattr(node, field, None)
            if value is None or (isinstance(value, (list, tuple)) and not value):
                continue
            parts.append(f"{field}={_stable_dump(value)}")
        return f"{type(node).__name__}({', '.join(parts)})"
    if isinstance(node, (list, tuple)):
        return "[" + ", ".join(_stable_dump(value) for value in node) + "]"
    return repr(node)


@functools.lru_cache(maxsize=None)
def module_behavior_hash(module_name):
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return None
    try:
        with open(path, encoding="utf-8") as source:
            tree = ast.parse(source.read(), filename=path)
        return hashlib.sha1(_stable_dump(_strip_docstrings(tree)).encode()).hexdigest()[:16]
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def validation_store():
    from nfsdict import NfsDict
    base_dir = os.environ.get(
        "RC_VALIDATE_CACHE_DIR",
        os.path.join(user_cache_dir("reasoning_core"), "validation"),
    )
    return NfsDict(name="examples", base_dir=base_dir, serializer="json")


def _parquet_safe(value):
    import pandas as pd
    try:
        pd.DataFrame([value]).to_parquet(BytesIO(), index=False)
        return True
    except Exception:
        return False


def serialize(data):
    if _parquet_safe(data):
        return data
    return "b64:" + base64.b64encode(pickle.dumps(data)).decode()


def deserialize(value):
    if isinstance(value, str) and value.startswith("b64:"):
        return pickle.loads(base64.b64decode(value[4:].encode()))
    return value


class TimeoutException(TimeoutError):
    pass


_RETRYABLE = (TimeoutException, subprocess.SubprocessError)


def timeout_retry(seconds=15, attempts=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            on_main = threading.current_thread() is threading.main_thread()
            if not on_main:
                warnings.warn(
                    "timeout_retry: signal-based timeout unavailable off the main thread; "
                    "call will run without a timeout guard.",
                    stacklevel=3,
                )

            def handler(signum, frame):
                module = frame.f_globals.get("__name__", "") if frame else ""
                if module == "ctypes" or module.startswith("z3"):
                    signal.alarm(1)
                    return
                raise TimeoutException()

            for attempt in range(1, attempts + 1):
                if on_main:
                    old_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    converted = isinstance(exc, ctypes.ArgumentError) and "TimeoutException" in str(exc)
                    if not isinstance(exc, _RETRYABLE) and not converted:
                        raise
                    if on_main:
                        signal.alarm(0)
                    try:
                        children = _psutil().Process().children(recursive=True)
                        for child in children:
                            child.kill()
                        _psutil().wait_procs(children, timeout=1)
                    except Exception:
                        pass
                    if attempt == attempts:
                        if converted:
                            raise TimeoutException() from exc
                        raise
                    time.sleep(0.5)
                finally:
                    if on_main:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


@functools.lru_cache(maxsize=1)
def load_tokenizer():
    cache_dir = user_cache_dir("reasoning_core")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache_dir)

    class WhitespaceTokenizer:
        def encode(self, text):
            return str(text).split()

    try:
        return _tiktoken().get_encoding("o200k_base")
    except Exception:
        return WhitespaceTokenizer()
