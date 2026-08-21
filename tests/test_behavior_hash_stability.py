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


@pytest.mark.parametrize("a,b", [
    ("x = 'a'\n", 'x = "a"\n'),                                   # quote style
    ("f(a)\n", "f(a,)\n"),                                        # trailing comma
    ("x = a + b\n", "x = (a + b)\n"),                             # redundant parens
    ("x = a + b\n", "x = (a +\n     b)\n"),                       # line wrapping
    ("x = a + b\n", "x = a + \\\n    b\n"),                       # backslash continuation
    ("def f():\n    return 1\n", "def f():\n  return 1\n"),       # indent width
    ("def f():\n    return 1\n", "def f():\n\treturn 1\n"),       # tabs vs spaces
    ("x = 1000\n", "x = 1_000\n"),                                # numeric separator
    ("x = 16\n", "x = 0x10\n"),                                   # literal base
    ('x = "ab"\n', 'x = "a" "b"\n'),                              # implicit concatenation
    ('x = "A"\n', 'x = "\\x41"\n'),                                # escape form
    ("x = 1\n", "x = 1  # why\n"),                                # inline comment
    ("x = 1\ny = 2\n", "x = 1\n\n\n\ny = 2\n"),                    # blank lines
    ("import os\n", "import os  # noqa: F401\n"),                 # linter pragma
    ('def f():\n    """A."""\n    return 1\n',
     'def f():\n    """Totally different text."""\n    return 1\n'),  # docstring wording
    ("class C:\n    x = 1\n", 'class C:\n    """Doc."""\n    x = 1\n'),
    ("x = 1\n", '"""Module doc."""\nx = 1\n'),
])
def test_semantically_empty_formatting_is_ignored(a, b):
    """Reformatting a task must not read as a rework: the hash gates re-measurement, and a build
    costs a day of fleet time. Anything a formatter or a linter pragma can change is invisible."""
    assert h(a) == h(b)


def test_canonicalisation_is_frozen():
    """A canary digest, so changing `_stable_dump` cannot happen by accident.

    Every recorded behavior_hash -- in cache manifests, in shipped rows, in the drift tooling -- is
    only comparable to others produced by the SAME canonicalisation. Changing it invalidates all of
    that evidence and costs a full regeneration, so it has to be a deliberate act with the constant
    below updated in the same commit, not a silent side effect of an unrelated edit.
    """
    assert h(BASE) == "2cbd8a747b60cdf0"
