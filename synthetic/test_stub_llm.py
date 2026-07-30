"""Test-only: monkeypatch litlm so llm_generate_tasks.py can be exercised
end-to-end without a real network / API key. Run tasks with small
--n-per-level values to sanity check the whole pipeline before spending
real money.

    PYTHONPATH=. python3 scripts/_test_stub_llm.py --task code_execution --levels 0 1 --n-per-level 2
"""
import itertools
import random
import re
import sys

import litlm

_counter = itertools.count()

STUB_PROGRAMS = [
    "def f0(a: int, b: int) -> int:\n    total = a\n    for i in range(b):\n        total += i\n    return total\n\n"
    "def endpoint(x: int, y: int) -> int:\n    return f0(x, y)\n",
    "def f0(s: str) -> str:\n    out = ''\n    for ch in s:\n        if ch == 'a':\n            out += 'b'\n        else:\n            out += ch\n    return out\n\n"
    "def f1(a: list) -> list:\n    return a[1:] if a else a\n\n"
    "def endpoint(s: str, a: list) -> str:\n    return f0(s) + str(len(f1(a)))\n",
    "def f0(a: list, i: int) -> int:\n    if i < 0:\n        i = -i\n    return a[i % max(1, len(a))]\n\n"
    "def endpoint(a: list, i: int) -> int:\n    return f0(a, i)\n",
]

STEP_BODIES = [
    "if flag:\n    count = (count + 1) % 4\nelse:\n    flag = random.choice([False, True])\n",
    "match mode:\n    case 'idle':\n        mode = 'busy'\n    case 'busy':\n        mode, level = 'done', level + 1\n    case _:\n        mode = random.choice(['idle', 'busy'])\n",
]

INT_MUTATORS = [
    "def f0(s: int) -> int:\n    if s % 3 == 0:\n        return s // 3 + 7\n    return s * 2 - 5\n",
    "def f0(s: int) -> int:\n    total = s\n    for i in range(3):\n        total = (total * 2 - i) % 97\n    return total\n",
]

TOOLKIT_BATCH = [
    {"name": "load_x", "inputs": [["path", "str"]], "output": "Blob"},
    {"name": "shape_of", "inputs": [["b", "Blob"]], "output": "tuple"},
    {"name": "transform", "inputs": [["b", "Blob"], ["scale", "float"]], "output": "Blob"},
    {"name": "to_list", "inputs": [["b", "Blob"]], "output": "RecordBatch"},
    {"name": "summarize_batch", "inputs": [["batch", "RecordBatch"]], "output": "Summary"},
    {"name": "render_summary", "inputs": [["summary", "Summary"], ["verbose", "bool"]], "output": "str"},
    {"name": "fit_model", "inputs": [["data", "Blob"], ["lr", "float"]], "output": "Model"},
    {"name": "predict", "inputs": [["model", "Model"], ["data", "Blob"]], "output": "Prediction"},
    {"name": "score_prediction", "inputs": [["pred", "Prediction"], ["truth", "Blob"]], "output": "float"},
    {"name": "save_model", "inputs": [["model", "Model"], ["path", "str"]], "output": "bool"},
    {"name": "log_metric", "inputs": [["name", "str"], ["value", "float"]], "output": "LogHandle"},
    {"name": "flush_log", "inputs": [["handle", "LogHandle"]], "output": "bool"},
]


class FakeText(str):
    def __new__(cls, content, call_id):
        obj = super().__new__(cls, content)
        obj.failed = False
        obj.call_id = call_id
        obj.cost = 0.0001
        return obj


def _reliteral(value):
    """The real prompt now renders each domain value with correct Python literal
    syntax (bare ints/bools, quoted strings), so ast.literal_eval already hands
    us a properly-typed Python value -- repr() alone reproduces valid code for
    any of int/bool/str without needing to guess from string content."""
    return repr(value)


def _stub_step_body_for(text):
    """Build a step-body stub that actually uses the real variable names AND
    domains the rig picked for this attempt (parsed out of the prompt via
    ast.literal_eval), rather than a hardcoded name set that would just
    NameError or violate a domain. Matches "  name: one of [...]" lines
    directly wherever they appear, rather than anchoring on specific
    surrounding prose (which changes as the real prompt wording is tuned)."""
    import ast
    vars_info = []
    for line in text.splitlines():
        mm = re.match(r"\s*(\w+): one of (\[.*\])\s*$", line)
        if mm:
            try:
                vars_info.append((mm.group(1), ast.literal_eval(mm.group(2))))
            except (ValueError, SyntaxError):
                continue
    if not vars_info:
        return random.choice(STEP_BODIES)
    stmts = [
        f"{name} = random.choice([{', '.join(_reliteral(v) for v in domain)}])"
        for name, domain in vars_info
    ]
    return "\n".join(stmts) + "\n"


def fake_complete(inputs, model=None, max_tokens=None, temperature=None, system=None,
                   json=False, show_progress=False, **kwargs):
    call_id = next(_counter)
    text = inputs if isinstance(inputs, str) else str(inputs)
    if "JSON array" in text:
        content = __import__("json").dumps(random.sample(TOOLKIT_BATCH, len(TOOLKIT_BATCH)))
    elif "step() body" in text or "step body" in text.lower():
        content = _stub_step_body_for(text)
    elif "f0(s: int)" in text:
        content = random.choice(INT_MUTATORS)
    elif "f0(x: int)" in text:
        content = "def f0(x: int) -> int:\n    if x < 0:\n        return -x + 3\n    return (x * 3) % 17\n"
    else:
        content = random.choice(STUB_PROGRAMS)
    return FakeText(f"```python\n{content}\n```" if "def " in content else content, call_id)


def fake_cost_breakdown(period="session"):
    return {"stub-model": 0.0001 * next(itertools.count(1))}


litlm.complete = fake_complete
litlm.cost_breakdown = fake_cost_breakdown
litlm.extract_json = lambda s, default=None: __import__("json").loads(s) if s.strip().startswith("[") else (default or [])

if __name__ == "__main__":
    import llm_generate_tasks
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    llm_generate_tasks.main()