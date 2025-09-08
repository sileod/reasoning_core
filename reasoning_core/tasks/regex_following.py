
import random, re
from pathlib import Path
from functools import cache
import string
import exrex
import regex
from dataclasses import dataclass
from unigram import init_grammar, generate
from reasoning_core.template import Task, Problem, register_dataset, Reward, Config
from easydict import EasyDict as edict
from faker import Faker
import sys, os
from functools import wraps
import codecs

#import re2 as re

def shutup(f):
    @wraps(f)
    def wrapper(*a, **kw):
        with open(os.devnull, 'w') as devnull:
            old = sys.stdout
            sys.stdout = devnull
            try: return f(*a, **kw)
            finally: sys.stdout = old
    return wrapper


fake = Faker()

wordlist = fake.words(nb=100,unique=True)

def regex_grammar():
    R = init_grammar(['re'], preprocess_template=lambda x: x)

    R('start(regex)', '{0}')
    R('regex(regex,regex)', '{0}{1}', weight=2)
    R('regex(regex)', '({0})', weight=2)
    R('regex(regex,regex)', '{0}|{1}', weight=1)
    R('regex(char)', '{0}',weight=1)
    R('regex(word)', '{0}',weight=1)

    for w in random.sample(wordlist, 8):
        R('word', w)

    R('regex(regex)?', '{0}?')
    R('regex(regex)*', '{0}*')
    R('regex(regex)+', '{0}+')
    R('regex(rangechar,rangechar)', '[{0}-{1}]')
    R('regex(predef)', '{0}',weight=3)

    chars = string.ascii_letters + string.digits
    for c in chars:
        R('char', c)
        R('rangechar', c)

    for s in [r'\d', r'\w', r'\s', '.', r'\.']:
        R('predef', s, weight=1)

    #for s in [r'\D', r'\W', r'\S', r'\\', r'\(', r'\)', r'\[', r'\]', r'\t', r'\n']:
    for s in [r'\D', r'\W', r'\S']: # Keep only non-matching classes if desired, but they can also be complex

        R('predef', s, weight=0.25)

    return R



@shutup
def safe_regex(r, max_tries=10, timeout_seconds=0.5):
    try:
        p = regex.compile(r)
    except regex.error:
        return False 
    for _ in range(max_tries):
        try:
            s = exrex.getone(r, 5)
            if s and p.fullmatch(s, timeout=timeout_seconds):
                return True 
        except TimeoutError:
            return False
        except Exception:
            continue
    return False

def sample_regex(config, max_tries=100):
    max_depth = config.max_depth
    min_depth = config.max_depth
    G = regex_grammar()
    for _ in range(max_tries):
        x = generate(G.start(), depth=max_depth, min_depth=min_depth)
        if len(x.leaves)<=1:
            continue
        r = x @ 're'
        if safe_regex(r):
            return r
    raise RuntimeError("No valid regex found")



@dataclass
class RegexConfig(Config):
    n_ex: int = 8
    max_depth: int = 6
    min_depth: int = 4

    def update(self, c):
        self.n_ex += c
        self.max_depth += c
        self.min_depth += c

@shutup
def sample_instance(r_str, max_tries=100):
    """Generates a non-empty string that is verified by re.fullmatch()."""
    try:
        #p = re.compile(r_str)
        p = regex.compile(r_str)

    except re.error:
        raise ValueError(f"Could not compile invalid regex: {r_str}")

    for _ in range(max_tries):
        s = exrex.getone(r_str, 5)
        # Verify the generated string is a non-empty full match
        if s and p.fullmatch(s, timeout=5):
            return s
    raise ValueError(f"Could not generate a verified string for regex: {r_str}")

class RegexFollowing(Task):
    def __init__(self, config=RegexConfig()):
        super().__init__(config=config)

    def generate(self):
        meta = edict()
        r = sample_regex(self.config)
        meta.regex = r
        meta.string = sample_instance(r)
        return Problem(meta, meta.string)


    def score_answer(self, answer, entry):
        try:
            answer_str, pattern = str(answer), entry['metadata']['regex']
            max_edits = len(answer_str) + len(pattern)
            
            distance = next((e for e in range(min(max_edits, 10) + 1)
                            if regex.fullmatch(f'(?:{pattern}){{e<={e}}}', answer_str, timeout=0.5)),
                            max_edits) # Corrected parenthesis here
                            
            return 1.0 / (1.0 + distance)
        
        except (TimeoutError, regex.error):
            return None

    def prompt(self, meta):
        return (
            "'daf' is a valid match for regex '[a-z]{3}' but not 'ab1'\n"
            f"Return a valid match for {meta.regex}"
        )

    def balancing_key(self, problem):
        return problem.metadata.regex

def strip_anchors_safe(text: str) -> str:
    """Strips optional ^ and non-escaped $ from a regex string."""
    # This is the robust one-liner
    m = regex.match(r"^\^?(.*?)(?<!\\)\$?$", text)
    return m.group(1) if m else text


class RegexInduction(Task):
    def __init__(self, config=RegexConfig()):
        super().__init__(config=config)

    def generate(self):
        meta = edict()
        meta.regex =sample_regex(self.config)
        meta.positives = [sample_instance(meta.regex) for _ in range(self.config.n_ex)]
        
        negatives = []
        while len(negatives) < self.config.n_ex:
            # Ensure negative examples do not match the target regex
            s = sample_instance(sample_regex(self.config))
            if not regex.fullmatch(meta.regex, s, timeout=1):
                negatives.append(s)
        meta.negatives = negatives
        
        return Problem(meta, meta.regex)

    def score_answer(self, answer, entry):
        predicted_regex, meta = str(answer), entry.metadata
        try:
            predicted_regex = strip_anchors_safe(predicted_regex)
            r = regex.compile(predicted_regex)
        except Exception as e: #HOW TO CATCH THAT MORE SPECIFICALLY
            return 0.0

        # Calculate success rates for positives and negatives separately.
        pos_rate = sum(bool(r.fullmatch(s)) for s in meta['positives']) / len(meta['positives'])
        neg_rate = sum(not r.fullmatch(s) for s in meta['negatives']) / len(meta['negatives'])
        # Score is the product of rates, ensuring 0 if no positives match.
        accuracy = pos_rate * neg_rate
        
        # Apply length penalty only on a perfect accuracy score (accuracy == 1.0).
        return min(1.0, len(meta['regex']) / (len(predicted_regex) or 1e-9)) if accuracy == 1.0 else accuracy

    def prompt(self, meta):
        pos_examples = ', '.join(f"'{s}'" for s in meta['positives'])
        neg_examples = ', '.join(f"'{s}'" for s in meta['negatives'])
        return (
            f"Return a regex that matches all POSITIVE strings and none of the NEGATIVE strings.\n"
            f"POSITIVE: {pos_examples}\n"
            f"NEGATIVE: {neg_examples}"
        )

