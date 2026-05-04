
import random, re
from pathlib import Path
import string
import exrex
import regex
from dataclasses import dataclass
from gramforge import init_grammar, generate
from reasoning_core.template import Task, Problem, register_dataset, Reward, Config
from easydict import EasyDict as edict
from faker import Faker
import sys, os
from functools import wraps
import codecs

#import re2 as re
r"""
ROADMAP:
Explicit Quantifiers ({n}, {n,m})
Explicit Character Sets ([abc])
Negated Character Classes ([^a-z], [^abc])
Escaped Literals (\+, \*, \?, etc.)
Non-Capturing Groups ((?:...))
"""


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


def regex_grammar(fsm_subset=False, alpha=None, words=None):
    R = init_grammar(["re"], preprocess_template=lambda x: x)

    R("start(regex)", "{0}")
    R("regex(regex,regex)", "{0}{1}", weight=2)
    R("regex(regex)", "({0})", weight=2)
    R("regex(regex,regex)", "{0}|{1}", weight=1)
    R("regex(char)", "{0}", weight=1)
    R("regex(word)", "{0}", weight=1)

    if fsm_subset: #greenery
        assert alpha and words
        R("regex(regex)?", "{0}?")
        R("regex(regex)*", "{0}*")
        R("regex(regex)+", "{0}+")
        for w in words: R("word", w)
        for c in alpha: R("char", c)
        return R

    for w in random.sample(wordlist, 8):
        R("word", w)

    R("regex(regex)?", "{0}?")
    R("regex(regex)*", "{0}*")
    R("regex(regex)+", "{0}+")

    for i in range(1, 4):
        R("count_exact", "{{%d}}" % i)
        for j in range(i + 1, 6):
            R("count_range", "{{%d,%d}}" % (i, j))

    R("regex(regex,count_exact)", "{0}{1}")
    R("regex(regex,count_range)", "{0}{1}")

    R("regex(rangechar,rangechar)", "[{0}-{1}]")
    R("regex(rangechar,rangechar)", "[^{0}-{1}]")
    R("regex(char,char,char)", "[{0}{1}{2}]")
    R("regex(char,char,char)", "[^{0}{1}{2}]")

    R("regex(regex)", "(?:{0})")

    R("regex(predef)", "{0}", weight=3)

    for c in string.ascii_letters + string.digits:
        R("char", c)
        R("rangechar", c)

    for s in [r"\d", r"\w", ".", r"\."]:
        R("predef", s, weight=1)

    for s in [r"\D", r"\W"]:
        R("predef", s, weight=0.25)

    for s in [r"\+", r"\*", r"\?", r"\\", r"\(", r"\)", r"\[", r"\]"]:
        R("predef", s, weight=0.25)

    return R


@shutup
def safe_regex(r):
    try:
        sample_instance(r, max_tries=10)
        return True
    except (ValueError, Exception):
        return False

def sample_regex(config, max_tries=100):
    max_depth = config.max_depth
    min_depth = config.min_depth

    G = regex_grammar()
    for _ in range(max_tries):
        x = generate(G.start(), depth=max_depth, min_depth=min_depth, mode=config.gramforge_algorithm)
        if len(x.leaves)<=1:
            continue
        r = x @ 're'
        if safe_regex(r):
            return r
    raise RuntimeError("No valid regex found")



@dataclass
class RegexConfig(Config):
    n_ex: int = 8
    max_depth: int = 5
    min_depth: int = 3
    gramforge_algorithm = "sequential"
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
        # Verify the generated string is a non-empty full match and has no unprintable characters
        if s and s.isprintable() and p.fullmatch(s, timeout=5):
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
            expected_len = len(entry['metadata']['string'])
            target_len_penalty = abs(len(answer_str) - expected_len)
            
            max_edits = len(answer_str) + len(pattern)
            
            distance = next((e for e in range(min(max_edits, 10) + 1)
                            if regex.fullmatch(f'(?:{pattern}){{e<={e}}}', answer_str, timeout=0.5)),
                            max_edits) # Corrected parenthesis here
                            
            return 1.0 / (1.0 + distance + target_len_penalty)
        
        except (TimeoutError, regex.error):
            return None

    def prompt(self, meta):
        n = len(meta.string)
        return f"The answer is a {n}-character string that fully matches the regular expression: {meta.regex}"

    def balancing_key(self, problem):
        return problem.metadata.regex

def strip_anchors_safe(text: str) -> str:
    """Strips optional ^, non-escaped $, and markdown formatting from a regex string."""
    if "```" in text:
        m = regex.search(r"```(?:regex|re|text)?\n(.*?)\n```", text, regex.DOTALL)
        if m: text = m.group(1)
    text = text.strip('\r\n').strip('`').strip('\r\n')
    m = regex.match(r"^\^?(.*?)(?<!\\)\$?$", text)
    return m.group(1) if m else text


class RegexInduction(Task):
    def __init__(self, config=RegexConfig()):
        super().__init__(config=config)

    def generate(self):
        while True:
            meta = edict()
            meta.regex = sample_regex(self.config)
            assert meta.regex == meta.regex.strip('\r\n').strip('`').strip('\r\n'), f"Gold regex incompatible with strip: {repr(meta.regex)}"
            positives = set()
            for _ in range(self.config.n_ex * 5):
                positives.add(sample_instance(meta.regex))
                if len(positives) == self.config.n_ex:
                    break
            if len(positives) < 2:
                continue
            meta.positives = list(positives)
            break
        
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
        except (regex.error, TimeoutError): # <--- CATCHING SPECIFICALLY
            return 0.0

        pos_rate = sum(bool(r.fullmatch(s)) for s in meta['positives']) / len(meta['positives'])
        neg_rate = sum(not r.fullmatch(s) for s in meta['negatives']) / len(meta['negatives'])
        
        accuracy = pos_rate * neg_rate
        
        if accuracy == 1.0:
            # Base score of 0.5 for perfect accuracy, plus up to 0.5 for being short
            length_ratio = len(meta['regex']) / max(1, len(predicted_regex))
            length_bonus = min(1.0, length_ratio) * 0.5
            return 0.5 + length_bonus
        else:
            return accuracy * 0.49

    def prompt(self, meta):
        pos_examples = ', '.join(f"'{s}'" for s in meta['positives'])
        neg_examples = ', '.join(f"'{s}'" for s in meta['negatives'])
        return (
            f"The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.\n"
            f"POSITIVE: {pos_examples}\n"
            f"NEGATIVE: {neg_examples}"
        )




from greenery import parse as gparse

ALPHA = "abcdefgh"



def _sample_pair(G, depth, min_depth, mode, max_tries=40):
    """Sample two non-equivalent regexes."""
    r1, f1 = _sample_regex(G, depth, min_depth, mode)
    if f1 is None:
        return None
    for _ in range(max_tries):
        r2, f2 = _sample_regex(G, depth, min_depth, mode)
        if f2 is not None and not f1.equivalent(f2):
            return r1, f1, r2, f2
    return None


def _shortest_witness(fsm):
    """Shortest string in fsm, or None if empty."""
    for s in fsm.strings(otherchars=[]):
        return s
    return None


@dataclass
class RegexReasoningConfig(Config):
    max_depth: int = 4
    min_depth: int = 2
    n_alpha: int = 3
    gramforge_algorithm: str = "sequential"

    def update(self, c):
        self.max_depth += c
        self.min_depth += c
        self.n_alpha += 0.5 * c



def _sample_regex(G, depth, min_depth, mode="sequential", max_tries=60):
    for _ in range(max_tries):
        x = generate(G.start(), depth=depth, min_depth=min_depth, mode=mode)
        if len(x.leaves) <= 1:
            continue
        r = x @ "re"
        try:
            f = gparse(r).to_fsm()
            if not f.empty():
                return r, f
        except Exception:
            continue
    return None, None

class RegexReasoning(Task):
    def __init__(self, config=RegexReasoningConfig()):
        super().__init__(config=config)

    def generate(self):
        cfg = self.config
        alpha = ALPHA[: max(2, cfg.n_alpha)]
        words = [a + b for a in alpha for b in alpha][:6]
        G = regex_grammar(fsm_subset=True, alpha=alpha, words=random.sample(words, min(len(words), 4)))

        pair = _sample_pair(G, cfg.max_depth, cfg.min_depth, cfg.gramforge_algorithm)
        if pair is None:
            return None
        r1, f1, r2, f2 = pair

        qtype = random.choice(["equivalence", "containment", "distinguishing"])

        if qtype == "equivalence":
            # ~40% "Yes" (reuse same regex string), ~60% "No" (use the pair)
            if random.random() < 0.4:
                meta = edict(qtype="equivalence", regex_a=r1, regex_b=r1)
                return Problem(meta, "Yes")
            meta = edict(qtype="equivalence", regex_a=r1, regex_b=r2)
            return Problem(meta, "No")

        if qtype == "containment":
            # Force ~50% Yes by building a superset via union
            if random.random() < 0.5:
                sup = gparse(f"({r1})|({r2})")
                r_sup = str(sup)
                # A=r1 ⊆ B=r1|r2 is always true
                meta = edict(qtype="containment", regex_a=r1, regex_b=r_sup)
                return Problem(meta, "Yes")
            else:
                is_sub = f1.issubset(f2)
                if random.random() < 0.5:
                    meta = edict(qtype="containment", regex_a=r1, regex_b=r2)
                    return Problem(meta, "Yes" if is_sub else "No")
                else:
                    meta = edict(qtype="containment", regex_a=r2, regex_b=r1)
                    return Problem(meta, "Yes" if f2.issubset(f1) else "No")

        # distinguishing
        sd = f1.symmetric_difference(f2)
        witness = _shortest_witness(sd)
        if witness is None:
            return None
        meta = edict(qtype="distinguishing", regex_a=r1, regex_b=r2)
        return Problem(meta, witness)

    def prompt(self, metadata):
        a, b = metadata["regex_a"], metadata["regex_b"]
        qt = metadata["qtype"]
        if qt == "equivalence":
            return (
                f"Consider the regular expressions A = {a} and B = {b}\n"
                f"Do A and B accept exactly the same set of strings?\n"
                f"The answer is Yes or No."
            )
        elif qt == "containment":
            return (
                f"Consider the regular expressions A = {a} and B = {b}\n"
                f"Is every string accepted by A also accepted by B?\n"
                f"The answer is Yes or No."
            )
        else:
            return (
                f"Consider the regular expressions A = {a} and B = {b}\n"
                f"Find the shortest string that is accepted by exactly one of A or B (but not both).\n"
                f"The answer is the shortest such string."
            )

    def score_answer(self, answer, entry):
        qt = entry.metadata["qtype"]
        answer = str(answer).strip()
        if answer.lower() in ("ε", "\\epsilon", "ε (the empty string)", '""', "''"):
            answer = ""
        if qt in ("equivalence", "containment"):
            norm = answer.lower().strip().rstrip(".")
            return float(norm == entry.answer.lower()) if norm in ("yes", "no") else 0.0
        # distinguishing: verify witness semantically
        try:
            fa = gparse(entry.metadata["regex_a"]).to_fsm()
            fb = gparse(entry.metadata["regex_b"]).to_fsm()
        except Exception:
            return 0.0
        if fa.accepts(answer) == fb.accepts(answer):
            return 0.0
        expected_len = len(entry.answer)
        return 1.0 / (1.0 + max(0, len(answer) - expected_len))

    def balancing_key(self, problem):
        return f"{problem.metadata.qtype}_{problem.answer}"
