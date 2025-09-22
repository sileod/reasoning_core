from unigram import init_grammar, generate
from tqdm.auto import tqdm
from functools import cache
from nltk.parse.generate import generate as nltk_generate
from nltk import CFG, ChartParser 
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

fake = Faker()


wordlist = list(fake.words(nb=500,unique=True))

from dataclasses import dataclass

@dataclass
class GrammarConfig(Config):
    n_types: int = 4
    n_terminals: int = 5
    perturbation_rate: float = 0.5

    min_depth:int =8
    max_depth:int =12

    min_prod_depth:int=5
    max_prod_depth:int=8



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



def sample_cfg(config=GrammarConfig):
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
        lambda _: (generate(nltk_to_unigram(sample_cfg(config)).get_rules('s', shuffle=True)[0], depth=5) @ 'lang').split()
    ])(tokens)

def generate_parse(config=GrammarConfig):
    meta = edict()
    while True:
        g = sample_cfg(config)
        g_u = nltk_to_unigram(g)
        rule = g_u.get_rules("s", shuffle=True)[0]
        try:
            tokens = (generate(rule, depth=config.max_prod_depth, min_depth = config.min_prod_depth) @ "lang").split()
        except ValueError:
            continue
        if random.random() < config.perturbation_rate:
            tokens = perturb(tokens, config)
        try:
            with timeout(2):
                meta.parses = [str(x) for x in list(ChartParser(g).parse(tokens))]
        except TimeoutError:
            continue
        except ValueError:
            meta.parses = None

        meta.label = ("unparsable" if not meta.parses else 
                 "ambiguous"   if len(meta.parses) > 1 else 
                 "unambiguous")
        meta.tokens = tokens
        meta.g = str(g).split('\n',1)[-1].strip()
        return meta


class Parsability(Task):
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        super().__init__(config=config)

    def generate(self):
        meta = generate_parse(self.config)
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
            label, *trees = meta.label, meta.parses
            if label == 'unambiguous':
                parse = " ".join(str(trees[0][0]).split())
                return Problem(meta, parse)


    def prompt(self, meta):
        g, tokens = meta.g, meta.tokens
        example = """Given G_ex: S -> NP VP, NP -> 'det' Noun, Noun -> 'noun', VP -> 'verb' \
        and G_ex: "det noun verb" correct Lisp Parse Tree would be (S (NP det (Noun noun)) (VP verb))."
        """
        return (
            f"(GRAMMAR)\n{g}\n\n"
            f"(STRING)\n{' '.join(tokens)}\n\n"
            f"(QUESTION)\n"
            "Return the fully parenthesized parse tree of STRING in Lisp style.\n"
            "Use uppercase for nonterminals, lowercase unquoted tokens for terminals\n"
            f"{example}"
        )

    def score_answer(self, answer, entry):
        reference = entry['answer']
        norm_space = lambda s: re.sub(r'\s+', ' ', s)
        prepr = lambda x: norm_space(str(x).strip()).replace('"','').replace("'",'')
        dist = edit_distance(prepr(answer), prepr(reference))
        return 1 / (1 + dist / (len(reference)**0.5 + 1))   