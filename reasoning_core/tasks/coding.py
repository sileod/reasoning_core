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
            if output is not None and len(output) > 0 and len(output) < 10:
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
    

import difflib
import random
import re
import hashlib
from dataclasses import dataclass
from easydict import EasyDict as edict
from faker import Faker
from rapidfuzz.distance import Levenshtein

# Assumed external dependencies (Reasoning Core)
from reasoning_core.template import Task, Problem, Config, edict

fake = Faker()

def get_short_hash():
    """Generates a git-style short hash (7 chars)."""
    r = str(random.random()).encode('utf-8')
    return hashlib.sha1(r).hexdigest()[:7]

@dataclass
class DiffConfig(Config):
    min_versions: int = 2
    max_versions: int = 5
    nb_lines: int = 5
    mutation_rate: float = 0.2

    def update(self, c):
        self.max_versions += c
        self.nb_lines += c

def mutate_words_in_line(line, vocab, rate):
    """Mutates words within a single string (line)."""
    words = line.split()
    if not words: return line
    
    if random.random() > rate:
        return line

    new_words = []
    for word in words:
        r = random.random()
        if r < 0.05:   # Delete word
            continue
        elif r < 0.15: # Substitute word
            new_words.append(random.choice(vocab))
        else:
            new_words.append(word)
    
    if not new_words and words:
        new_words = [random.choice(vocab)]
        
    return " ".join(new_words)

def mutate_lines(lines, vocab, rate):
    """Evolves a list of lines (sentences)."""
    new_lines = []
    
    for line in lines:
        r = random.random()
        if r < rate / 5:       # Delete entire line
            continue
        elif r < rate:         # Modify line
            new_lines.append(mutate_words_in_line(line, vocab, rate)) 
        elif r < rate * 1.2:   # Insert new line
            new_lines.append(" ".join(fake.words(nb=5)))
            new_lines.append(line)
        else:
            new_lines.append(line)
            
    if not new_lines:
        new_lines.append(" ".join(fake.words(nb=5)))
        
    return new_lines

def get_git_diff(src_lines, tgt_lines):
    """Generates a standard Git-style unified diff without file headers."""
    diff = difflib.unified_diff(src_lines, tgt_lines, lineterm='')
    # Strip the first two lines (--- and +++) to leave only chunks
    return "\n".join(list(diff)[2:])

class VersionedTask:
    def __init__(self, config=DiffConfig()):
        super().__init__(config=config)
        self.vocab = list(fake.words(nb=500, unique=True))

    def generate_version_chain(self):
        lines = [fake.sentence(nb_words=6).rstrip('.') for _ in range(self.config.nb_lines)]
        vid = get_short_hash()
        
        chain = [{'id': vid, 'lines': lines}]

        n_versions = random.randint(self.config.min_versions, self.config.max_versions)
        for _ in range(n_versions - 1):
            prev_lines = chain[-1]['lines']
            new_lines = mutate_lines(prev_lines, self.vocab, self.config.mutation_rate)
            new_vid = get_short_hash()
            chain.append({'id': new_vid, 'lines': new_lines})
            
        return chain

    def select_pair(self, chain):
        idxs = list(range(len(chain)))
        i = random.choice(idxs)
        j = random.choice([x for x in idxs if x != i])
        return chain[i], chain[j]

class DiffPrediction(VersionedTask, Task):
    def generate(self):
        chain = self.generate_version_chain()
        src, tgt = self.select_pair(chain)
        
        # Diff without headers
        diff_str = get_git_diff(src['lines'], tgt['lines'])
        
        history_text = []
        for v in chain:
            content = "\n".join(v['lines'])
            history_text.append(f"Version {v['id']}:\n{content}\n")

        meta = edict(
            history="\n".join(history_text),
            src_id=src['id'],
            tgt_id=tgt['id']
        )
        return Problem(meta, diff_str)

    def prompt(self, meta):
        return (f"Below is the version history of a file.\n\n"
                f"{meta.history}\n"
                f"Generate the Unified Diff to transform version {meta.src_id} into version {meta.tgt_id}.\n"
                f"Answer with the diff chunks only (no file headers). If no changes, return nothing.")

    def score_answer(self, answer, entry):
        return Levenshtein.normalized_similarity(answer.strip(), entry['answer'].strip())

class DiffPatching(VersionedTask, Task):
    def generate(self):
        chain = self.generate_version_chain()
        src, tgt = self.select_pair(chain)
        
        diff_str = get_git_diff(src['lines'], tgt['lines'])
        
        meta = edict(
            src_text="\n".join(src['lines']),
            src_id=src['id'],
            tgt_id=tgt['id'],
            diff=diff_str
        )
        return Problem(meta, "\n".join(tgt['lines']))

    def prompt(self, meta):
        return (f"Apply the following Unified Diff to the text.\n\n"
                f"Original Text (Version {meta.src_id}):\n"
                f"{meta.src_text}\n\n"
                f"Diff ({meta.src_id} -> {meta.tgt_id}):\n"
                f"{meta.diff}\n\n"
                f"Answer with the resulting text only.")

    def score_answer(self, answer, entry):
        return Levenshtein.normalized_similarity(answer.strip(), entry['answer'].strip())