# nltk must be fully initialized before gramforge, which imports nltk internally
import nltk  # noqa: F401
from nltk.parse.generate import generate as nltk_generate
from nltk import CFG, ChartParser
from nltk.parse.earleychart import EarleyChartParser
from nltk.data import path as nltk_path
from nltk.tree import Tree
from nltk.grammar import CFG, Nonterminal
from gramforge import init_grammar, generate_with_choices
from gramforge import generate as gramforge_generate
from gramforge.grammars import simple_english_grammar, arith_grammar, dyck_grammar
from gramforge import gramforge_to_nltk
from tqdm.auto import tqdm
from functools import cache
from contextlib import contextmanager
import sys
from reasoning_core.template import Task, Problem, Config
import random
from pathlib import Path
import string
from easydict import EasyDict as edict
from faker import Faker
import re
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from itertools import islice, combinations


fake = Faker()

existing_grammars = [
    simple_english_grammar(), simple_english_grammar(questions=False),
    dyck_grammar(), dyck_grammar(include_unicode=False)
]
existing_grammars = [gramforge_to_nltk(g) for g in existing_grammars]

wordlist = list(fake.words(nb=500,unique=True))

from dataclasses import dataclass

@dataclass
class GrammarConfig(Config):
    n_types: int = 4
    n_terminals: int = 5
    perturbation_rate: float = 0.5

    gramforge_algorithm="sequential"
    min_depth:int =5
    max_depth:int =8

    min_prod_depth:int=4
    max_prod_depth:int=6

    random_grammar_prob:float = 0.3
    tagging_prob: float = 0.5
    target_num_rules=10

    n_resampled_grammars: int=200
    prob_resampling_grammar: float=0.6

    def update(self, c):
        self.n_types += c
        self.n_terminals += c
        self.min_depth += c
        self.max_depth += c
        self.prob_resampling_grammar = max(0.0, self.prob_resampling_grammar - 0.1 * c)

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

def nltk_to_gramforge(g):
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



def prune_cfg(grammar):
    prods = list(grammar.productions())

    by_lhs = defaultdict(list)
    for p in prods:
        by_lhs[p.lhs()].append(p)

    # reachable from start
    reachable = {grammar.start()}
    stack = [grammar.start()]
    while stack:
        lhs = stack.pop()
        for p in by_lhs.get(lhs, []):
            for s in p.rhs():
                if isinstance(s, Nonterminal) and s not in reachable:
                    reachable.add(s)
                    stack.append(s)

    prods = [p for p in prods if p.lhs() in reachable]

    # productive NTs
    productive = set()
    changed = True
    while changed:
        changed = False
        for p in prods:
            if all((isinstance(s, str) or s in productive) for s in p.rhs()):
                if p.lhs() not in productive:
                    productive.add(p.lhs())
                    changed = True

    if grammar.start() not in productive:
        return None

    prods = [
        p for p in prods
        if p.lhs() in productive
        and all((isinstance(s, str) or s in productive) for s in p.rhs())
    ]

    return CFG(grammar.start(), prods)

def sample_cfg(config=GrammarConfig, productive_only=False):
    if random.random() > config.random_grammar_prob:
        g = random.choice(existing_grammars)
        if len(g.productions()) > config.target_num_rules:
            g = trim_grammar(g, config.target_num_rules)
        if productive_only:
            g = prune_cfg(g)
            if g is None:
                raise ValueError("Existing grammar became unproductive")
        return g

    for _ in range(1000):
        MG = meta_grammar(config).start()
        for _ in range(100):
            x = gramforge_generate(MG, depth=config.max_depth, min_depth=config.min_depth, mode=config.gramforge_algorithm)
            try:
                g = CFG.fromstring(x@"cfg")
            except ValueError:
                continue

            if productive_only:
                g = prune_cfg(g)
                if g is None:
                    continue

            try:
                prods = list(islice(nltk_generate(g, depth=config.max_prod_depth), 10))
            except (RecursionError, ValueError):
                continue

            if len(prods) > 3:
                return g

    raise ValueError("Failed to sample CFG")

@contextmanager
def resampled_grammar(config, **kw):
    if random.random() < config.prob_resampling_grammar:
        seed = random.randint(0, config.n_resampled_grammars - 1)
        state = random.getstate()
        try:
            random.seed(seed)
            yield sample_cfg(config, productive_only=True)
        finally:
            random.setstate(state)
    else:
        yield sample_cfg(config, **kw)

def perturb(tokens, config=GrammarConfig):
    return random.choice([
        lambda t: random.sample(t, len(t)),
        lambda t: (lambda i: t[:i]+t[i+1:])(random.randrange(len(t))) if len(t)>1 else t,
        #lambda _: (gramforge_generate(nltk_to_unigram(sample_cfg(config)).get_rules('s', shuffle=True)[0], depth=5, mode=config.gramforge_algorithm) @ 'lang').split()
        lambda _: (gramforge_generate(nltk_to_gramforge(sample_cfg(config)), depth=5, mode=config.gramforge_algorithm) @ 'lang').split()

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

    return "\n".join(lines), ps

def generate_parse(config=GrammarConfig):
    meta = edict()
    while True:
        with resampled_grammar(config) as g:
            g_u = nltk_to_gramforge(g)
            
            try:
                tokens = (gramforge_generate(g_u, depth=config.max_prod_depth, min_depth=config.min_prod_depth, mode=config.gramforge_algorithm) @ "lang").split()
            except ValueError: continue

            if random.random() < config.perturbation_rate:
                tokens = perturb(tokens, config)

            try:
                meta.cot, meta.parses = make_cot(g, tokens)
            except (RecursionError, ValueError):
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
            f"The answer is exactly one word: unambiguous, ambiguous, or unparsable."
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

            t = meta.parses[0] # Get the Tree object directly

            if random.random() < self.config.tagging_prob:
                meta.mode = 'tagging'
                leaves = []
                for idx in t.treepositions('leaves'):
                    token = t[idx]
                    pos = t[idx[:-1]].label() # Parent label
                    depth = len(idx)          # Distance from root
                    leaves.append(f"{token}<{pos}:{depth}>")
                return Problem(meta, " ".join(leaves))
            else:
                meta.mode = 'parsing'
                tree_str = " ".join(str(t).split())
                return Problem(meta, tree_str)

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
            "The answer is the fully parenthesized parse tree of STRING in Lisp style.\n"
            f"{ex}")


    def score_answer(self, answer, entry):
        norm = lambda s: re.sub(r'\s+', ' ', str(s).strip()).replace('"','').replace("'",'')

        reference = entry['answer']
        if not answer: return 0.0
        
        return Levenshtein.normalized_similarity(norm(answer), norm(reference))


def _edge_str(edge):
    rhs = [str(s) for s in edge.rhs()]
    dot = edge.dot()
    rhs = rhs[:dot] + ['•'] + rhs[dot:]
    return f"{edge.lhs()}→{' '.join(rhs)}"

def get_valid_next_tokens(grammar, prefix):
    """
    Exact next-token oracle for prefix-safe grammars:
    - valid next terminals
    - whether STOP is valid
    - lightweight edge-based justifications
    """
    parser = EarleyChartParser(grammar)
    nullable, first = _compute_nullable_and_first(grammar)

    try:
        chart = parser.chart_parse(list(prefix))
    except ValueError:
        return set(), False, {}

    n = len(prefix)
    valid_tokens = set()
    justifications = {}
    can_stop = False

    for edge in chart.select(end=n):
        edge_txt = _edge_str(edge)

        if edge.is_complete():
            if edge.start() == 0 and edge.lhs() == grammar.start():
                can_stop = True
                justifications.setdefault("STOP", edge_txt)
            continue

        remainder = edge.rhs()[edge.dot():]
        toks, _ = _first_of_sequence(remainder, first, nullable)

        for tok in toks:
            valid_tokens.add(tok)
            justifications.setdefault(tok, edge_txt)

    return valid_tokens, can_stop, justifications


def _build_cot(tokens, can_stop, justifications):
    parts = []

    if can_stop and 'STOP' in justifications:
        parts.append(f"{justifications['STOP']}⇒STOP")

    grouped = defaultdict(list)
    for tok in sorted(tokens):
        grouped[justifications.get(tok, "continuation")].append(tok)

    for reason, toks in sorted(grouped.items()):
        if len(toks) > 3:
            parts.append(f"{reason}⇒{{{','.join(toks)}}}")
        else:
            parts.extend(f"{reason}⇒{tok}" for tok in toks)

    return "\n".join(parts) if parts else "continuation"


class Continuation(Task):
    """Grammar continuation task using proper CFG parsing."""
    
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        super().__init__(config=config)
        self.balancing_key_ratio = 0.1
        
    def generate(self):
        for _ in range(100):
            with resampled_grammar(self.config, productive_only=True) as g:
                try:
                    sentences = list(islice(nltk_generate(g, depth=self.config.max_depth), 50))
                    if not sentences:
                        continue
                    sentence = random.choice(sentences)
                except (RecursionError, ValueError):
                    continue
                
                if len(sentence) < 2:
                    continue
                
                max_prefix = min(len(sentence) - 1, 5)
                min_prefix = min(2, max_prefix)
                if min_prefix > max_prefix:
                    continue
                prefix_len = random.randint(min_prefix, max_prefix)
                prefix = list(sentence[:prefix_len])
                
                try:
                    tokens, can_stop, justifications = get_valid_next_tokens(g, prefix)
                except Exception:
                    continue
                
                if not tokens and not can_stop:
                    continue
                
                answer = '|'.join(sorted(tokens))
                if can_stop:
                    answer = (answer + '|STOP') if answer else 'STOP'
                
                cot = _build_cot(tokens, can_stop, justifications)
                
                return Problem(
                    edict(g="\n".join(str(p) for p in g.productions()), 
                          prefix=prefix, depth=len(prefix), cot=cot),
                    answer
                )
        raise ValueError("Failed to generate continuation after 100 attempts")
    
    def prompt(self, meta):
        pfx = ' '.join(meta.prefix) if meta.prefix else '<empty>'
        return (f"List all valid next tokens for this prefix. "
                f"The answer is the list of valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.\n"
                f"(GRAMMAR)\n{meta.g}\n(PREFIX)\n{pfx}")

    def score_answer(self, answer, entry):
        prepr = lambda x: {e.strip() for e in x.split('|')}
        try:
            ref, ans = prepr(entry['answer']), prepr(answer)
            inter = len(ref & ans)
            # Jaccard
            return inter / max(len(ref | ans), 1)
        except Exception:  # also: bare except catches KeyboardInterrupt
            return 0


# --- Error Detection Task ---

def _span_hits(seq, span):
    k = len(span)
    return [i for i in range(len(seq) - k + 1) if seq[i:i+k] == span]

def min_context(tokens, idx):
    for start in range(idx, -1, -1):
        if len(_span_hits(tokens, tokens[start:idx+1])) == 1:
            return ' '.join([*tokens[start:idx], f'>>{tokens[idx]}<<'])
    return ' '.join([*tokens[:idx], f'>>{tokens[idx]}<<'])

def grammar_terminals(g):
    return sorted({s for p in g.productions() for s in p.rhs() if isinstance(s, str)})

def prefix_has_completion(g, prefix):
    valid, can_stop, _ = get_valid_next_tokens(g, prefix)
    return can_stop or bool(valid)

def first_error_marked(g, tokens):
    lines = []
    for i, tok in enumerate(tokens):
        prev_valid, _, _ = get_valid_next_tokens(g, tokens[:i])

        if not prefix_has_completion(g, tokens[:i+1]):
            lines.append(f"{tok} ∉ {{{','.join(sorted(prev_valid)[:8])}}}")
            ans = min_context(tokens, i)
            lines.append(f"Answer: {ans}")
            return ans, i, '\n'.join(lines)

        lines.append(f"{tok} ✓")

    _, can_stop, _ = get_valid_next_tokens(g, tokens)
    return ('OK' if can_stop else 'INCOMPLETE'), -1, '\n'.join(lines)

def corrupt_once(g, tokens):
    terms = grammar_terminals(g)
    if len(terms) < 2:
        raise ValueError("Need ≥2 terminals")
    for _ in range(80):
        op = random.choices(['substitute', 'insert', 'delete'], weights=[6, 2, 2])[0]
        if op == 'delete' and len(tokens) < 3:
            continue
        if op == 'delete':
            pos = random.randrange(len(tokens))
            out = tokens[:pos] + tokens[pos+1:]
        elif op == 'insert':
            pos = random.randrange(len(tokens) + 1)
            valid, _, _ = get_valid_next_tokens(g, tokens[:pos])
            bad = [t for t in terms if t not in valid]
            if not bad:
                continue
            out = tokens[:pos] + [random.choice(bad)] + tokens[pos:]
        else:  # substitute
            pos = random.randrange(len(tokens))
            valid, _, _ = get_valid_next_tokens(g, tokens[:pos])
            # prefer terminals valid elsewhere but invalid here
            bad = [t for t in terms if t not in valid and t != tokens[pos]]
            if not bad:
                continue
            out = list(tokens)
            out[pos] = random.choice(bad)

        answer, idx, cot = first_error_marked(g, out)
        if answer not in ('OK', 'INCOMPLETE'):
            return out, answer, idx, cot
    raise ValueError("Failed to corrupt")

def get_marked_index(toks):
    for i, t in enumerate(toks):
        if t.startswith('>>') and t.endswith('<<'):
            return i
    return -1

def _norm_marked(s):
    s = re.sub(r'>>\s+', '>>', str(s).strip())
    s = re.sub(r'\s+<<', '<<', s)
    return re.sub(r'\s+', ' ', s)

class LocateError(Task):
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        config.perturbation_rate = 0.0
        super().__init__(config=config)

    def generate(self):
        for _ in range(100):
            with resampled_grammar(self.config, productive_only=True) as g:
                if len(grammar_terminals(g)) < 2:
                    continue
                try:
                    toks = (gramforge_generate(
                        nltk_to_gramforge(g),
                        depth=self.config.max_prod_depth,
                        min_depth=self.config.min_prod_depth,
                        mode=self.config.gramforge_algorithm
                    ) @ "lang").split()
                except ValueError:
                    continue
                if len(toks) < 3:
                    continue

                roll = random.random()
                if roll < 0.15:
                    ans, idx, cot = first_error_marked(g, toks)
                    if ans != 'OK':
                        continue
                    out = toks
                elif roll < 0.30:
                    out = toks[:random.randint(1, len(toks) - 1)]
                    ans, idx, cot = first_error_marked(g, out)
                    if ans != 'INCOMPLETE':
                        continue
                else:
                    try:
                        out, ans, idx, cot = corrupt_once(g, toks)
                    except ValueError:
                        continue

                return Problem(
                    edict(g="\n".join(str(p) for p in g.productions()),
                          tokens=out, error_index=idx, cot=cot),
                    ans
                )
        raise ValueError("Failed to generate locate-error task")

    def prompt(self, meta):
        return (
            f"(GRAMMAR)\n{meta.g}\n\n"
            f"(STRING)\n{' '.join(meta.tokens)}\n\n"
            f"The answer is the shortest contiguous span from STRING that ends at the first invalid token "
            f"and occurs only once in STRING.\n"
            f"Mark the invalid token as >>token<<.\n"
            f"If the token alone is enough, answer just >>token<<.\n"
            f"If STRING is fully grammatical, answer OK.\n"
            f"If all shown tokens are valid but more are needed, answer INCOMPLETE.\n"
            f"One line only."
        )

    def score_answer(self, answer, entry):
            if not answer: return 0.0
            a, r = _norm_marked(answer), _norm_marked(entry['answer'])
            if a == r: return 1.0
            if {'OK', 'INCOMPLETE'} & {a, r}: return 0.0

            a_toks = a.split()
            marked = [t.startswith('>>') and t.endswith('<<') for t in a_toks]
            if marked.count(True) != 1 or not marked[-1]:
                return 0.0

            span = [t.replace('>>', '').replace('<<', '') for t in a_toks]
            hits = _span_hits(entry.metadata['tokens'], span)
            
            if entry.metadata['error_index'] not in {h + len(span) - 1 for h in hits}:
                return 0.0

            # Strict penalty: the span is correct but occurs multiple times in the text
            if len(hits) > 1:
                return 0.5 

            # Unambiguous location: minimum 0.9, scales to 1.0 based on how concise the prefix is
            efficiency = len(r.split()) / len(a_toks)
            return min(1.0, max(0.9, efficiency))


# --- Constrained Generation ---

# --- Constrained Generation ---


def _compute_nullable_and_first(grammar):
    """Exact nullable + FIRST sets via fixed-point iteration (no depth cutoff)."""
    nts = {p.lhs() for p in grammar.productions()}

    nullable = set()
    changed = True
    while changed:
        changed = False
        for p in grammar.productions():
            rhs = p.rhs()
            if not rhs or all(
                isinstance(s, Nonterminal) and s in nullable for s in rhs
            ):
                if p.lhs() not in nullable:
                    nullable.add(p.lhs())
                    changed = True

    first = {nt: set() for nt in nts}
    changed = True
    while changed:
        changed = False
        for p in grammar.productions():
            add, _ = _first_of_sequence(p.rhs(), first, nullable)
            before = len(first[p.lhs()])
            first[p.lhs()].update(add)
            if len(first[p.lhs()]) != before:
                changed = True

    return nullable, first


def _first_of_sequence(seq, first, nullable):
    """FIRST terminals reachable from a symbol sequence + whether it's all-nullable."""
    out = set()
    all_nullable = True
    for sym in seq:
        if isinstance(sym, str):
            out.add(sym)
            all_nullable = False
            break
        out.update(first.get(sym, set()))
        if sym not in nullable:
            all_nullable = False
            break
    return out, all_nullable


def _exact_next_tokens_and_stop(grammar, prefix, parser=None, nullable=None, first=None):
    """
    Sound next-token discovery via Earley boundary edges + exact FIRST/nullable.
    Returns (valid_tokens: set[str], can_stop: bool).
    """
    parser = parser or EarleyChartParser(grammar)
    if nullable is None or first is None:
        nullable, first = _compute_nullable_and_first(grammar)

    try:
        chart = parser.chart_parse(list(prefix))
    except ValueError:
        return set(), False

    n = len(prefix)
    valid_tokens = set()
    can_stop = False

    for edge in chart.select(end=n):
        if edge.is_complete():
            if edge.start() == 0 and edge.lhs() == grammar.start():
                can_stop = True
            continue
        remainder = edge.rhs()[edge.dot():]
        toks, _ = _first_of_sequence(remainder, first, nullable)
        valid_tokens.update(toks)

    return valid_tokens, can_stop


def exact_completions(grammar, prefix, k, max_states=4096):
    """
    All distinct k-length suffixes making prefix+suffix a complete sentence.
    Returns [] (safe skip) if state space overflows — never produces wrong results.
    """
    prefix = list(prefix)
    parser = EarleyChartParser(grammar)
    nullable, first = _compute_nullable_and_first(grammar)

    frontier = {()}
    for _ in range(k):
        nxt = set()
        for suf in frontier:
            toks, _ = _exact_next_tokens_and_stop(
                grammar, prefix + list(suf), parser, nullable, first
            )
            for tok in toks:
                nxt.add(suf + (tok,))
        if not nxt or len(nxt) > max_states:
            return []
        frontier = nxt

    return [
        list(suf)
        for suf in sorted(frontier)
        if _exact_next_tokens_and_stop(
            grammar, prefix + list(suf), parser, nullable, first
        )[1]
    ]


def find_blanked_target(candidates, min_blanks=2, max_blanks=3):
    """
    Pick a target and a fill-in-the-blanks template.

    Start with ALL positions revealed (hinted), then greedily remove hints
    (creating blanks) while the target remains the unique candidate that
    matches every remaining hint. Randomised removal order for variety.

    Returns (target_as_list, hints {pos: token}) or None.
    """
    cands = [tuple(c) for c in candidates]
    k = len(cands[0])
    max_blanks = min(max_blanks, k - 1)          # keep ≥1 hint
    if min_blanks > max_blanks:
        return None

    order = list(cands)
    random.shuffle(order)

    for target in order:
        hinted = set(range(k))
        positions = list(range(k))
        random.shuffle(positions)

        for pos in positions:
            if k - len(hinted) >= max_blanks:    # enough blanks
                break
            trial = hinted - {pos}
            alive = sum(
                1 for c in cands if all(c[i] == target[i] for i in trial)
            )
            if alive == 1:                       # still unique → blank it
                hinted = trial

        n_blanks = k - len(hinted)
        if min_blanks <= n_blanks <= max_blanks:
            return list(target), {i: target[i] for i in sorted(hinted)}

    return None


def _format_template(k, hints):
    return " ".join(hints.get(i, "___") for i in range(k))


def _blanked_cot(prefix, candidates, target, hints, k):
    blanks = sorted(set(range(k)) - set(hints.keys()))
    lines = [
        f"{len(candidates)} valid {k}-token continuations",
        f"Template: {_format_template(k, hints)}",
        f"Blanks at positions: {blanks}",
    ]

    alive = [tuple(c) for c in candidates]
    for i in sorted(hints.keys()):
        prev = len(alive)
        alive = [c for c in alive if c[i] == hints[i]]
        if len(alive) < prev:
            lines.append(
                f"  pos[{i}]='{hints[i]}': {prev} → {len(alive)} candidates"
            )

    for b in blanks:
        vals = sorted({c[b] for c in alive})
        lines.append(f"  pos[{b}] options: {{{', '.join(vals)}}}")

    if len(alive) <= 6:
        for j, c in enumerate(alive, 1):
            mark = " ✓" if tuple(c) == tuple(target) else ""
            lines.append(f"  {j}. {' '.join(c)}{mark}")

    lines.append(f"Answer: {' '.join(target)}")
    return "\n".join(lines)


class ConstrainedContinuation(Task):
    """Fill-in-the-blanks grammar continuation.

    Given a grammar, a prefix, and a mostly-revealed continuation
    template, find the unique completion that forms a grammatical
    sentence.  Revealed tokens are fixed; ___ marks blanks to fill.
    """

    def __init__(
        self,
        config: GrammarConfig = GrammarConfig(),
        min_k=3,
        max_k=5,
        min_blanks=2,
        max_blanks=3,
        max_options=20,
    ):
        super().__init__(config=config)
        self.min_k = max(3, min_k)
        self.max_k = max_k
        self.min_blanks = min_blanks
        self.max_blanks = max_blanks
        self.max_options = max_options
        self.balancing_key_ratio = 0.1

    def generate(self):
        for _ in range(200):
            with resampled_grammar(self.config, productive_only=True) as g:
                try:
                    sentences = [
                        list(s)
                        for s in islice(
                            nltk_generate(g, depth=self.config.max_depth), 80
                        )
                        if len(s) >= self.min_k + 1
                    ]
                except (RecursionError, ValueError):
                    continue
                if not sentences:
                    continue

                sent = random.choice(sentences)
                max_plen = min(5, len(sent) - self.min_k)
                if max_plen < 1:
                    continue

                for plen in random.sample(range(1, max_plen + 1), max_plen):
                    prefix = sent[:plen]
                    ks = list(range(
                        self.min_k,
                        min(self.max_k, len(sent) - plen) + 1,
                    ))
                    random.shuffle(ks)

                    for k in ks:
                        cands = exact_completions(g, prefix, k)
                        if not (2 <= len(cands) <= self.max_options):
                            continue

                        result = find_blanked_target(
                            cands, self.min_blanks, self.max_blanks
                        )
                        if result is None:
                            continue

                        target, hints = result
                        blanks = sorted(set(range(k)) - set(hints.keys()))

                        # Safety: verify uniqueness among exact candidates
                        alive = [
                            c for c in cands
                            if all(c[i] == hints[i] for i in hints)
                        ]
                        if len(alive) != 1 or alive[0] != target:
                            continue

                        return Problem(
                            edict(
                                g="\n".join(str(p) for p in g.productions()),
                                k=k,
                                prefix=prefix,
                                hints={str(i): tok for i, tok in hints.items()},
                                template=_format_template(k, hints),
                                blanks=blanks,
                                n_blanks=len(blanks),
                                n_hints=len(hints),
                                n_options=len(cands),
                                cot=_blanked_cot(
                                    prefix, cands, target, hints, k
                                ),
                            ),
                            " ".join(target),
                        )

        raise ValueError(
            "Failed to generate constrained continuation after 200 attempts"
        )

    def prompt(self, meta):
        pfx = " ".join(meta.prefix) if meta.prefix else "<empty>"
        nb = meta.n_blanks
        bw = "blank" if nb == 1 else "blanks"
        return (
            f"(GRAMMAR)\n{meta.g}\n\n"
            f"(PREFIX)\n{pfx}\n\n"
            f"(TEMPLATE)\n{meta.template}\n\n"
            f"Fill in the {nb} {bw} (___) to form a grammatical continuation "
            f"of PREFIX using exactly {meta.k} tokens.\n"
            f"Fixed tokens must remain in place. "
            f"The answer is all {meta.k} tokens space-separated."
        )

    def score_answer(self, answer, entry):
        if not answer:
            return 0.0

        ans = answer.strip().split()
        ref = entry["answer"].split()

        if ans == ref:
            return 1.0
        if len(ans) != len(ref):
            return 0.0

        # Revealed positions are hard constraints stated in the prompt
        hints = entry.metadata["hints"]
        for i, tok in hints.items():
            idx = int(i)
            if idx >= len(ans) or ans[idx] != tok:
                return 0.0

        # Partial credit based on blank accuracy
        blanks = [int(b) for b in entry.metadata["blanks"]]
        if not blanks:
            return 0.0
        blank_correct = sum(1 for b in blanks if ans[b] == ref[b]) / len(blanks)

        # Grammaticality bonus
        try:
            g = CFG.fromstring(entry.metadata["g"])
            _, can_stop = _exact_next_tokens_and_stop(
                g, list(entry.metadata["prefix"]) + ans
            )
        except Exception:
            can_stop = False

        if can_stop:
            return 0.3 + 0.6 * blank_correct   # 0.3 – 0.9
        return 0.15 * blank_correct             # 0.0 – 0.15