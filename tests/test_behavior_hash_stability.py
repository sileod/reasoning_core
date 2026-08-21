"""behavior_hash must be identical across Python versions, and still track real edits.

The first property is why `_stable_dump` exists: `ast.dump` gained fields in 3.12, so a byte-identical
task file hashed differently on the generation fleet (3.10) than on a dev box (3.12), and every task
read as permanently drifted. The second property is what makes it worth having at all -- a hash that
never changes would satisfy the first requirement and be useless.

The third block is the risk `_stable_dump` introduces: it drops fields that are None or empty, so
constructs that differ ONLY by such a field must still be distinguished.
"""
import ast
import hashlib

import pytest

from reasoning_core.template import _stable_dump, _strip_docstrings

BASE = "def f(x):\n    y = 1\n    return x + y\n"


def h(src):
    return hashlib.sha1(_stable_dump(_strip_docstrings(ast.parse(src))).encode()).hexdigest()[:16]


@pytest.mark.parametrize("src", [
    "def f(x):\n    # hello\n    y = 1\n    return x + y\n",
    "def f(x):\n\n    y = 1\n\n    return x + y\n",
    'def f(x):\n    """Doc."""\n    y = 1\n    return x + y\n',
    "def f(x):   \n    y = 1   \n    return x + y\n",
])
def test_cosmetic_edits_do_not_change_the_hash(src):
    assert h(src) == h(BASE)


@pytest.mark.parametrize("src", [
    "def f(x):\n    y = 2\n    return x + y\n",          # constant
    "def f(x):\n    y = 1\n    return x - y\n",          # operator
    "def f(x):\n    z = 1\n    return x + z\n",          # name
    "def f(x):\n    return x + y\n    y = 1\n",          # statement order
    "def f(x, k):\n    y = 1\n    return x + y\n",       # signature
])
def test_real_edits_change_the_hash(src):
    assert h(src) != h(BASE)


@pytest.mark.parametrize("a,b", [
    ("def f():\n    return\n", "def f():\n    return None\n"),
    ("x = None\n", "x = 0\n"),
    ("x = None\n", "x = False\n"),
    ("f()\n", "f(a)\n"),
    ("x = []\n", "x = ()\n"),
    ("x = []\n", "x = {}\n"),
    ("def f(a):\n    pass\n", "def f(a=None):\n    pass\n"),
    ("class C:\n    pass\n", "class C(B):\n    pass\n"),
    ("try:\n    p()\nexcept:\n    q()\n", "try:\n    p()\nexcept E:\n    q()\n"),
])
def test_dropping_none_and_empty_fields_introduces_no_collisions(a, b):
    assert h(a) != h(b)


def test_dump_carries_no_version_specific_field_names():
    """`type_params` is the 3.12 addition that broke parity; nothing like it may reach the digest."""
    dump = _stable_dump(ast.parse("def f[T](x: T) -> T:\n    return x\n"
                                  if hasattr(ast, "TypeVar") else BASE))
    assert "type_params=[]" not in dump
    assert "type_comment=None" not in dump
