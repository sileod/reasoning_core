from reasoning_core.template import Task, Problem, Config, edict
from reasoning_core.utils import score_scalar
from unigram import generate
from unigram.grammars import tinypy_grammar
from nltk.metrics.distance import edit_distance
import re

import io
import sys
import contextlib
import random
from dataclasses import dataclass
from typing import List


@dataclass
class CodeExecutionCfg(Config):
    difficulty: float = 0.0  # Scales from 0 (mostly 1.1) to 6+ (mostly 4.1)
    min_depth: int = 4
    max_depth: int = 15
    max_attempts: int = 100

    def update(self, c):
        self.difficulty += c
        self.max_depth += int(c)

class CodeExecution(Task):
    VALID_LEVELS = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1"]

    def __init__(self, config=CodeExecutionCfg()):
        super().__init__(config=config)

    def _get_tinypy_level(self) -> str:
        # Weighted selection: center around the difficulty index
        n = len(self.VALID_LEVELS)
        target = min(n - 1, max(0, int(self.config.difficulty)))
        weights = [1.0 / (1.0 + abs(i - target) ** 2) for i in range(n)]
        return random.choices(self.VALID_LEVELS, weights=weights)[0]

    def _execute_code(self, code_str: str) -> str:
        # Capture stdout; returns None if runtime error occurs
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                exec(code_str, {'__builtins__': __builtins__}, {})
            return f.getvalue().strip()
        except Exception:
            return None

    def generate(self) -> Problem:
        for _ in range(self.config.max_attempts):
            level = self._get_tinypy_level()
            g = tinypy_grammar(level=level)
            
            # Generate code tree
            x = generate(g, depth=self.config.max_depth, min_depth=self.config.min_depth)
            code = x @ 'py'
            
            # Filter trivial code or invalid syntax/runtime errors
            if "print" not in code: continue
            
            output = self._execute_code(code)
            
            # We want valid execution that produces some output
            if output is not None and len(output) > 0 and len(output) < 1000:
                meta = edict(code=code, tinypy_level=level)
                return Problem(metadata=meta, answer=output)

        raise RuntimeError(f"Failed to generate valid code task. Config: {self.config}")

    def prompt(self, metadata: dict) -> str:
        return (
            f"Predict the printed output of the following Python code:\n\n"
            f"```python\n{metadata.code}\n```\n\n"
            f"Return only the exact printed output string."
        )

    def score_answer(self, answer, entry):
        reference = entry['answer']
        norm_space = lambda s: re.sub(r'\s+', ' ', s)
        prepr = lambda x: norm_space(str(x).strip()).replace('"','').replace("'",'')
        dist = edit_distance(prepr(answer), prepr(reference))
        return 1 / (1 + dist / (len(reference)**0.5 + 1))