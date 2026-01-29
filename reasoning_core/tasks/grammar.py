from networkx import edges
from unigram import init_grammar, generate
from tqdm.auto import tqdm
from functools import cache
from nltk.parse.generate import generate as nltk_generate
from nltk import CFG, ChartParser
from nltk.parse.earleychart import EarleyChartParser
import sys
from reasoning_core.template import Task, Problem, Config, register_dataset
import random
from pathlib import Path
from nltk.data import path as nltk_path
import string
from easydict import EasyDict as edict
from faker import Faker
import re
from timeoutcontext import timeout
from nltk.tree import Tree
from collections import defaultdict
from unigram.grammars import simple_english_grammar, arith_grammar
from unigram import unigram_to_nltk
from rapidfuzz.distance import Levenshtein
from itertools import islice
from nltk.grammar import CFG, Nonterminal


fake = Faker()

existing_grammars = [simple_english_grammar(),simple_english_grammar(questions=False), arith_grammar()]
existing_grammars = [unigram_to_nltk(g) for g in existing_grammars]

wordlist = list(fake.words(nb=500,unique=True))

from dataclasses import dataclass

@dataclass
class GrammarConfig(Config):
    n_types: int = 4
    n_terminals: int = 5
    perturbation_rate: float = 0.5

    min_depth:int =5
    max_depth:int =8

    min_prod_depth:int=4
    max_prod_depth:int=6

    random_grammar_prob:float = 0.3
    tagging_prob: float = 0.5
    target_num_rules=10

    def update(self, c):
        self.n_types += c
        self.n_terminals += c
        self.min_depth += c
        self.max_depth += c

def meta_grammar(config):
    R=init_grammar(['cfg'])
    R('start(grammar)', '0')
    R('grammar(nonterminal,rules)', 'S -> 0\n1')

    R('rules(rule)', '0')
    R('rules(rule,rules)', '0\n1')
    R('rules(rule,rule,rules)', '0\n1\n2')

    R('rule(nonterminal,rhs)', '0 -> 1')

    R('rhs(expr)', '0')

    R('expr(symbol)', '0')
    R('expr(symbol,expr)', '0 1')
    R('expr(expr,symbol)', '0 1')

    R('symbol(nonterminal)', '0')
    R('symbol(terminal)', '0')
    R('expr(dyck)','0')

    for x in string.ascii_uppercase[:config.n_types]:
        R('nonterminal', x)

    R('terminal(t_rnd)', '0')
    for x in random.sample(wordlist, config.n_terminals):
        R('t_rnd', f"'{x}'")

    paren_types = [
        ('square', '[', ']'), ('curly', '<', '>'),
    ]

    for name, open_char, close_char in paren_types:
        R('dyck(expr)', f"'{open_char}'0'{close_char}'")

    return R

def nltk_to_unigram(g):
    import nltk
    R = init_grammar(['lang'])
    for p in g.productions():
        lhs = str(p.lhs()).lower()
        args, tokens, idx = [], [], 0
        for sym in p.rhs():
            if isinstance(sym, nltk.grammar.Nonterminal):
                tokens.append(str(idx))
                args.append(str(sym).lower())
                idx += 1
            else:
                tokens.append(sym)
        sig = f"{lhs}({','.join(args)})" if args else lhs
        R(sig, ' '.join(tokens))
    return R


def trim_grammar(grammar, target_size=10, retries=10, shrink_tries=1000, seed=None, max_steps=10000):
    rng = random.Random(seed)

    by_lhs = defaultdict(list)
    for p in grammar.productions():
        by_lhs[p.lhs()].append(p)

    def get_new_deps(rule, defined):
        return [s for s in rule.rhs() if isinstance(s, Nonterminal) and s not in defined]

    def prune(prods):
        if not prods:
            return []

        # map for reachability walk
        local_map = defaultdict(list)
        for p in prods:
            local_map[p.lhs()].append(p)

        # 1) reachable
        reachable = {grammar.start()}
        stack = [grammar.start()]
        while stack:
            lhs = stack.pop()
            for p in local_map.get(lhs, []):
                for s in p.rhs():
                    if isinstance(s, Nonterminal) and s not in reachable:
                        reachable.add(s)
                        stack.append(s)

        prods = [p for p in prods if p.lhs() in reachable]

        # 2) productive (fixed point)
        productive = set()
        changed = True
        while changed:
            changed = False
            for p in prods:
                if p.lhs() in productive:
                    continue
                if all((not isinstance(s, Nonterminal)) or (s in productive) for s in p.rhs()):
                    productive.add(p.lhs())
                    changed = True

        if grammar.start() not in productive:
            return []

        # 3) drop rules that reference unproductive NTs
        return [p for p in prods
                if p.lhs() in productive and
                   all((not isinstance(s, Nonterminal)) or (s in productive) for s in p.rhs())]

    for _ in range(retries):
        kept = set()
        defined = set()
        pending = [grammar.start()]

        # --- PHASE 1: GROW ---
        steps = 0
        while steps < max_steps:
            steps += 1

            if pending:
                lhs = pending.pop()
                if lhs in defined:
                    continue
                options = by_lhs.get(lhs, [])
                if not options:
                    break
            elif len(kept) < target_size:
                expandable = [(l, [p for p in by_lhs[l] if p not in kept]) for l in defined]
                expandable = [(l, opts) for l, opts in expandable if opts]
                if not expandable:
                    break
                lhs, options = rng.choice(expandable)
            else:
                break

            if not options:
                continue

            # Improved near-budget selection: minimize number of NEW deps
            if len(kept) >= target_size:
                dep_counts = [(len(get_new_deps(p, defined)), p) for p in options]
                m = min(c for c, _ in dep_counts)
                options = [p for c, p in dep_counts if c == m]

            rule = rng.choice(options)
            kept.add(rule)
            defined.add(lhs)
            pending.extend(get_new_deps(rule, defined))

        # --- PHASE 2: SHRINK ---
        current = prune(list(kept))
        if not current:
            continue

        for _ in range(shrink_tries):
            if len(current) <= target_size:
                break
            cand = rng.choice(current)
            trial = [p for p in current if p != cand]
            trial = prune(trial)
            if trial:
                current = trial

        return CFG(grammar.start(), current)

    print(f"Warning: trimming failed after {retries} retries.")
    return grammar



def sample_cfg(config=GrammarConfig):
    if random.random()>config.random_grammar_prob:
        return trim_grammar(random.choice(existing_grammars), config.target_num_rules)
        
    for _ in range(1000):
        MG = meta_grammar(config).start()
        for _ in range(100): 
            x=generate(MG,depth=config.max_depth,min_depth=config.min_depth)
            g = CFG.fromstring(x@"cfg")
            try:
                with timeout(1):
                    prods=list(nltk_generate(g ,depth=config.max_prod_depth,n=10))
            except TimeoutError:
                continue
            if len(prods)>3:
                return g

def perturb(tokens, config=GrammarConfig):
    return random.choice([
        lambda t: random.sample(t, len(t)),
        lambda t: (lambda i: t[:i]+t[i+1:])(random.randrange(len(t))) if len(t)>1 else t,
        #lambda _: (generate(nltk_to_unigram(sample_cfg(config)).get_rules('s', shuffle=True)[0], depth=5) @ 'lang').split()
        lambda _: (generate(nltk_to_unigram(sample_cfg(config)), depth=5) @ 'lang').split()

    ])(tokens)

def make_cot(g, tokens):
    # Get up to 2 parses to detect ambiguity without exhaustively searching
    ps = list(islice(EarleyChartParser(g).parse(tokens), 2))
    
    lines = []
    for i, t in enumerate(ps, 1):
        lines.append(f"Parse {i}:")
        for idx in t.treepositions('leaves'):
            # Construct path: Root -> ... -> POS
            path = [t[idx[:k]].label() for k in range(len(idx))]
            lines.append(f"'{t[idx]}': {' > '.join(path)} (Depth: {len(path)})")


    return "\n".join(lines), [str(p) for p in ps]

def generate_parse(config=GrammarConfig):
    meta = edict()
    while True:
        g = sample_cfg(config)
        g_u = nltk_to_unigram(g)
        
        try:
            tokens = (generate(g_u, depth=config.max_prod_depth, min_depth=config.min_prod_depth) @ "lang").split()
        except ValueError: continue

        if random.random() < config.perturbation_rate:
            tokens = perturb(tokens, config)

        try:
            with timeout(2):
                meta.cot, meta.parses = make_cot(g, tokens)
        except (TimeoutError, ValueError):
            continue

        meta.label = ("unparsable" if not meta.parses else 
                     "ambiguous"   if len(meta.parses) > 1 else 
                     "unambiguous")
        meta.tokens = tokens
        meta.g = "\n".join(str(p) for p in g.productions())
        return meta


class Parsability(Task):
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        super().__init__(config=config)
        self.balancing_key_ratio=1/3

    def generate(self):
        meta = generate_parse(self.config)
        del meta['parses'] #can blow up_
        return Problem(meta, meta.label)

    def prompt(self, meta):
        g, tokens = meta.g, meta.tokens
        return (
            f"(GRAMMAR)\n{g}\n\n"
            f"(STRING)\n{' '.join(tokens)}\n\n"
            f"(QUESTION)\nWhat is the parsability of this string?\n"
            f"Answer with exactly one word, unambiguous|ambiguous|unparsable"
        )


class Parsing(Task):
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        config.perturbation_rate = 0.0
        super().__init__(config=config)

    def generate(self):
        while True:
            meta = generate_parse(self.config)
            if meta.label != 'unambiguous': continue
            meta.cot = meta.cot.split('\n',1)[1]  # Remove first line

            tree_str = meta.parses[0] # Get the Lisp-style string
            #meta.cot = make_tree_cot(meta.parses[0])
            if random.random() < self.config.tagging_prob:
                meta.mode = 'tagging'
                t = Tree.fromstring(tree_str)
                leaves = []
                for idx in t.treepositions('leaves'):
                    token = t[idx]
                    pos = t[idx[:-1]].label() # Parent label
                    depth = len(idx)          # Distance from root
                    leaves.append(f"{token}<{pos}:{depth}>")
                return Problem(meta, " ".join(leaves))
            else:
                meta.mode = 'parsing'
                return Problem(meta, " ".join(tree_str.split()))

    def prompt(self, meta):
        g, tokens = meta.g, meta.tokens
        head = f"(GRAMMAR)\n{g}\n\n(STRING)\n{' '.join(tokens)}\n\n(QUESTION)\n"
        
        if meta.mode == 'tagging':
            return (head + 
                "Identify the Part-of-Speech (immediate parent) and tree depth for each token.\n"
                "format per token: token<POS:depth>\n"
                "Example: the<Det:3> cat<Noun:3>")
        
        ex = """Given G_ex: S -> NP VP, NP -> 'd' N, N -> 'n', VP -> 'v' and "d n v", correct is (S (NP d (N n)) (VP v))."""
        return (head + 
            "Return the fully parenthesized parse tree of STRING in Lisp style.\n"
            f"{ex}")


    def score_answer(self, answer, entry):
        norm = lambda s: re.sub(r'\s+', ' ', str(s).strip()).replace('"','').replace("'",'')

        reference = entry['answer']
        if not answer: return 0.0
        
        return Levenshtein.normalized_similarity(norm(answer), norm(reference))