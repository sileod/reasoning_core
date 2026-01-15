from curses import meta
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
from nltk.metrics.distance import edit_distance
import re
from timeoutcontext import timeout
from nltk.tree import Tree
from collections import defaultdict
from unigram.grammars import simple_english_grammar, arith_grammar
from unigram import unigram_to_nltk

fake = Faker()

existing_grammars = [*[simple_english_grammar()]*2, arith_grammar()]
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

def drop_rules(grammar, frac=0.5):
    # Group by LHS
    by_lhs = defaultdict(list)
    for p in grammar.productions():
        by_lhs[p.lhs()].append(p)

    kept_prods = []
    for prods in by_lhs.values():
        # Guarantee at least one rule survives per LHS to maintain validity
        mandatory = random.choice(prods)
        candidates = [p for p in prods if p != mandatory]
        
        kept_prods.append(mandatory)
        kept_prods.extend(p for p in candidates if random.random() > frac)

    return CFG(grammar.start(), kept_prods)

def sample_cfg(config=GrammarConfig):
    if random.random()>config.random_grammar_prob:
        return drop_rules(random.choice(existing_grammars))
        
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
    header = "Action Span Rule\n"
    chart = EarleyChartParser(g).chart_parse(tokens)
    
    get_action = lambda e: "[SCAN]" if isinstance(e, str) else \
                           "[COMPLETE]" if e.is_complete() else \
                           "[PREDICT]" if e.dot() == 0 else "[ADVANCE]"

    # Filter out 0-length predictions (noise), keep progress & tokens
    edges = [e for e in chart.edges() if isinstance(e, str) or True]
    edges.sort(key=lambda e: (e.end(), e.length())) # Sort by locality

    cot = header + "\n".join(f"{get_action(e)} {e}" for e in edges)
    parses = [str(x) for x in chart.parses(g.start())]
    
    return cot, parses

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

            tree_str = meta.parses[0] # Get the Lisp-style string
            
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
        reference = entry['answer']
        norm = lambda s: re.sub(r'\s+', ' ', str(s).strip()).replace('"','').replace("'",'')
        dist = edit_distance(norm(answer), norm(reference))
        return 1 / (1 + dist / (len(reference)**0.5 + 1))