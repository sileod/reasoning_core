from gramforge import init_grammar, generate_with_choices
from gramforge import generate as gramforge_generate
from tqdm.auto import tqdm
from functools import cache
from contextlib import contextmanager
from nltk.parse.generate import generate as nltk_generate
from nltk import CFG, ChartParser
from nltk.parse.earleychart import EarleyChartParser
import sys
from reasoning_core.template import Task, DevTask, Entry, Config
import random
from pathlib import Path
from nltk.data import path as nltk_path
import string
from easydict import EasyDict as edict
from faker import Faker
import re
from collections import Counter, defaultdict
from gramforge.grammars import simple_english_grammar, arith_grammar, dyck_grammar
from gramforge import gramforge_to_nltk
from rapidfuzz.distance import Indel, Levenshtein
from itertools import islice
from nltk.grammar import CFG, Nonterminal, Production


fake = Faker()

existing_grammars = [
    simple_english_grammar(), simple_english_grammar(questions=False),
    dyck_grammar(), dyck_grammar(include_unicode=False)
]
existing_grammars = [gramforge_to_nltk(g) for g in existing_grammars]

wordlist = list(fake.words(nb=500,unique=True))

from dataclasses import dataclass

def grammar_text(grammar, config=None):
    rules = list(dict.fromkeys(map(str, grammar.productions())))
    random.shuffle(rules)
    if random.random() < getattr(config, "lean_style_prob", 0.0):
        rules = [rule.replace(" -> ", " := ", 1) for rule in rules]
    return "\n".join(rules)

@dataclass
class GrammarConfig(Config):
    n_types: int = 4
    n_terminals: int = 5
    perturbation_rate: float = 0.5
    lean_style_prob: float = 0.5

    gramforge_algorithm: str = "sequential"
    min_depth:int =5
    max_depth:int =8

    min_prod_depth:int=4
    max_prod_depth:int=6

    random_grammar_prob:float = 0.3
    free_form_grammar_prob: float = 0.05
    tagging_prob: float = 0.5
    target_num_rules: int = 10
    min_num_rules: int = 4
    max_num_rules: int = 8

    n_resampled_grammars: int=200
    prob_resampling_grammar: float=0.1
    max_tokens:int =16

    min_k: int = 3
    max_k: int = 5
    min_blanks: int = 2
    max_blanks: int = 3
    min_options: int = 4
    max_options: int = 25
    def apply_difficulty(self, level):
        self.n_types += level
        self.n_terminals += level
        self.min_depth += level
        self.max_depth += level
        self.min_num_rules += level
        self.max_num_rules += 2 * level
        self.prob_resampling_grammar = max(0.0, self.prob_resampling_grammar - 0.1 * level)
        self.max_tokens += 2 * level

def meta_grammar(config):
    R=init_grammar(['cfg'])
    R('start(grammar)', '0')
    R('grammar(nonterminal,rules)', 'S -> 0\n1')

    R('rules(rule)', '0')
    R('rules(rule,rules)', '0\n1')
    R('rules(rule,rule,rules)', '0\n1\n2')

    R('rule(nonterminal,rhs)', '0 -> 1', constraint=[lambda x: x[0] @ 0 != x[1] @ 0])

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

def random_productive_cfg(config=None):
    """Construct a small CFG whose active nonterminals are reachable and productive."""
    config = config or GrammarConfig()
    n_rules = random.randint(config.min_num_rules, config.max_num_rules)
    names = ["S", *(c for c in string.ascii_uppercase if c != "S")]
    n_nts = random.randint(2, min(config.n_types + 1, n_rules, len(names)))
    nts = [Nonterminal(name) for name in names[:n_nts]]
    terms = random.sample(wordlist, config.n_terminals)
    productions, seen = [], set()

    def add(lhs, rhs):
        production = Production(lhs, tuple(rhs))
        if production in seen:
            return
        seen.add(production)
        productions.append(production)

    # A chain makes every active nonterminal reachable and terminating.
    for lhs, child in zip(nts, nts[1:]):
        add(lhs, (child,))
    add(nts[-1], (random.choice(terms),))

    attempts = 0
    while len(productions) < n_rules and attempts < 1000:
        attempts += 1
        lhs = random.choice(nts)
        kind = random.choices(
            ["terminal", "mixed", "binary", "recursive"],
            weights=[3, 4, 2, 1],
        )[0]
        if kind == "terminal":
            rhs = (random.choice(terms),)
        elif kind == "mixed":
            nt, terminal = random.choice(nts), random.choice(terms)
            rhs = random.choice(((terminal, nt), (nt, terminal)))
        elif kind == "binary":
            rhs = (random.choice(nts), random.choice(nts))
        else:
            opening, closing = random.choice((("[", "]"), ("<", ">")))
            rhs = (opening, lhs, closing)
        add(lhs, rhs)

    if len(productions) != n_rules:
        raise RuntimeError(f"Could only construct {len(productions)}/{n_rules} productions")
    random.shuffle(productions)
    return CFG(nts[0], productions)


def _free_form_cfg(config, productive_only):

    for _ in range(1000):
        MG = meta_grammar(config).start()
        for _ in range(100):
            try:
                x = gramforge_generate(MG, depth=config.max_depth, min_depth=config.min_depth, mode=config.gramforge_algorithm)
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


def sample_cfg(config=None, productive_only=False):
    config = config or GrammarConfig()
    if random.random() < config.random_grammar_prob:
        if random.random() >= config.free_form_grammar_prob:
            return random_productive_cfg(config)
        return _free_form_cfg(config, productive_only)

    g = random.choice(existing_grammars)
    if len(g.productions()) > config.target_num_rules:
        g = trim_grammar(g, config.target_num_rules)
    if productive_only:
        g = prune_cfg(g)
        if g is None:
            raise ValueError("Existing grammar became unproductive")
    return g

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

def perturb(tokens, config=None):
    config = config or GrammarConfig()
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

def generate_parse(config=None, max_attempts=200):
    config = config or GrammarConfig()
    meta = edict()
    for _ in range(max_attempts):
        with resampled_grammar(config) as g:
            g_u = nltk_to_gramforge(g)
            
            try:
                tokens = (gramforge_generate(g_u, depth=config.max_prod_depth, min_depth=config.min_prod_depth, mode=config.gramforge_algorithm) @ "lang").split()
            except ValueError: continue

            if random.random() < config.perturbation_rate:
                try:
                    tokens = perturb(tokens, config)
                except ValueError:
                    continue
            
            if len(tokens) > config.max_tokens:
                continue

            try:
                meta.cot, meta.parses = make_cot(g, tokens)
            except (RecursionError, ValueError):
                continue

            meta.label = ("unparsable" if not meta.parses else 
                         "ambiguous"   if len(meta.parses) > 1 else 
                         "unambiguous")
            meta.tokens = tokens
            meta.g = grammar_text(g, config)
            meta.start = str(g.start())
            return meta
    raise RuntimeError(f"Failed to generate a parse after {max_attempts} attempts")


class Parsability(DevTask):
    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig())
        self.balancing_key_ratio=1/3

    def generate_entry(self):
        meta = generate_parse(self.config)
        del meta['parses'] #can blow up_
        return Entry(meta, meta.label)

    def render_prompt(self, meta):
        g, tokens = meta.g, meta.tokens
        return (
            f"(GRAMMAR)\n{g}\n\n"
            f"(STRING)\n{' '.join(tokens)}\n\n"
            f"(QUESTION)\nWhat is the parsability of this string?\n"
            f"The answer is exactly one word: unambiguous, ambiguous, or unparsable."
        )


class Parsing(DevTask):
    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig())
        self.config.perturbation_rate = 0.0

    def generate_entry(self):
        while True:
            meta = generate_parse(self.config)
            if meta.label != 'unambiguous': continue
            _, _, tail = meta.cot.partition('\n')
            if not tail: continue  # Skip if cot has no content after header
            meta.cot = tail

            t = meta.parses[0] # Get the Tree object directly

            if random.random() < self.config.tagging_prob:
                meta.mode = 'tagging'
                leaves = []
                for idx in t.treepositions('leaves'):
                    token = t[idx]
                    pos = t[idx[:-1]].label() # Parent label
                    depth = len(idx)          # Distance from root
                    leaves.append(f"{token}<{pos}:{depth}>")
                return Entry(meta, " ".join(leaves))
            else:
                meta.mode = 'parsing'
                tree_str = " ".join(str(t).split())
                return Entry(meta, tree_str)

    def render_prompt(self, meta):
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


def labeled_rules(meta):
    lines = list(dict.fromkeys(meta.g.splitlines()))
    random.shuffle(lines)
    return "\n".join(f"R{i}: {rule}" for i, rule in enumerate(lines)), {
        rule.replace(" := ", " -> ", 1): f"R{i}" for i, rule in enumerate(lines)
    }


class ParsingDerivation(Task):
    summary = "Determine the derivation production rule sequence parsing a given string."
    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig(
            target_num_rules=8, min_prod_depth=3, max_prod_depth=5, max_tokens=12,
        ))
        self.config.perturbation_rate = 0.0

    def generate_entry(self):
        for _ in range(200):
            meta = generate_parse(self.config)
            if meta.label != "unambiguous" or len(meta.parses) != 1:
                continue
            meta.labeled_g, labels = labeled_rules(meta)
            try:
                answer = " ".join(labels[str(p)] for p in meta.parses[0].productions())
            except KeyError:
                continue
            meta.pop("parses", None)
            meta.pop("cot", None)
            return Entry(meta, answer)
        raise RuntimeError("Failed to generate an unambiguous derivation after 200 attempts")

    def render_prompt(self, meta):
        return (
            f"(START)\n{meta.start}\n\n"
            f"(GRAMMAR)\n{meta.labeled_g}\n\n"
            f"(STRING)\n{' '.join(meta.tokens)}\n\n"
            "(QUESTION)\n"
            "The answer is the rule labels used in the leftmost derivation of STRING, in order, separated by spaces."
        )

    def score_answer(self, answer, entry):
        pred = re.findall(r"\bR\d+\b", str(answer))
        gold = entry["answer"].split()
        valid = set(re.findall(r"\bR\d+\b", entry.metadata["labeled_g"]))
        format_score = sum(rule in valid for rule in pred) / max(len(pred), len(gold))
        return .9 * Indel.normalized_similarity(pred, gold) + .1 * format_score



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


class Continuation(DevTask):
    """Grammar continuation task using proper CFG parsing."""
    
    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig())
        self.balancing_key_ratio = 0.1
        
    def generate_entry(self):
        for _ in range(100):
            with resampled_grammar(self.config, productive_only=True) as g:
                try:
                    sentence = (gramforge_generate(
                        nltk_to_gramforge(g),
                        depth=self.config.max_prod_depth,
                        min_depth=self.config.min_prod_depth,
                        mode=self.config.gramforge_algorithm,
                    ) @ "lang").split()
                except (RecursionError, ValueError):
                    continue
                
                min_len = 3 + self.config.level
                if len(sentence) < min_len or len(sentence) > self.config.max_tokens:
                    continue

                max_prefix = len(sentence) - 1
                min_prefix = min(max_prefix, 2 + self.config.level)
                cuts = list(range(min_prefix, max_prefix + 1))
                if len(cuts) > 6:
                    cuts = [min_prefix, max_prefix] + random.sample(cuts[1:-1], 4)
                candidates = []
                for prefix_len in cuts:
                    prefix = list(sentence[:prefix_len])
                    try:
                        tokens, can_stop, justifications = get_valid_next_tokens(g, prefix)
                    except Exception:
                        continue
                    n_answers = len(tokens) + int(can_stop)
                    if n_answers:
                        candidates.append((prefix_len + n_answers, prefix,
                                           tokens, can_stop, justifications))

                if not candidates:
                    continue

                preferred = [c for c in candidates
                             if len(c[2]) + int(c[3]) > 1]
                if self.config.level >= 3:
                    preferred = [c for c in preferred
                                 if len(c[2]) + int(c[3]) >= 3] or preferred
                if preferred and random.random() < 0.8:
                    candidates = preferred
                best = max(c[0] for c in candidates)
                _, prefix, tokens, can_stop, justifications = random.choice(
                    [c for c in candidates if c[0] == best]
                )
                
                answer = '|'.join(sorted(tokens))
                if can_stop:
                    answer = (answer + '|STOP') if answer else 'STOP'
                
                cot = _build_cot(tokens, can_stop, justifications)
                
                return Entry(
                    edict(g=grammar_text(g, self.config), start=str(g.start()),
                          prefix=prefix, depth=len(prefix), cot=cot),
                    answer
                )
        raise ValueError("Failed to generate continuation after 100 attempts")
    
    def render_prompt(self, meta):
        pfx = ' '.join(meta.prefix) if meta.prefix else '<empty>'
        return (f"List valid next tokens for this prefix. "
                f"The answer is the valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.\n"
                f"(START)\n{meta.start}\n(GRAMMAR)\n{meta.g}\n(PREFIX)\n{pfx}")

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

def grammar_terminals(g):
    return sorted({s for p in g.productions() for s in p.rhs() if isinstance(s, str)})

def first_error(g, tokens):
    counts = Counter()
    for i, tok in enumerate(tokens):
        counts[tok] += 1
        valid, _, _ = get_valid_next_tokens(g, tokens[:i])
        if tok not in valid:
            occurrence = f"@{counts[tok]}" if tokens.count(tok) > 1 else ""
            return f"ERROR {tok}{occurrence}", i

    _, can_stop, _ = get_valid_next_tokens(g, tokens)
    return ("OK" if can_stop else "INCOMPLETE"), -1

def corrupt_once(g, tokens):
    terms = grammar_terminals(g)
    weights = Counter(tokens)
    if len(terms) < 2:
        raise ValueError("Need ≥2 terminals")

    def choose(candidates):
        return random.choices(candidates, [weights[t] + 1 for t in candidates])[0]

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
            out = tokens[:pos] + [choose(bad)] + tokens[pos:]
        else:  # substitute
            pos = random.randrange(len(tokens))
            valid, _, _ = get_valid_next_tokens(g, tokens[:pos])
            # prefer terminals valid elsewhere but invalid here
            bad = [t for t in terms if t not in valid and t != tokens[pos]]
            if not bad:
                continue
            out = list(tokens)
            out[pos] = choose(bad)

        answer, idx = first_error(g, out)
        if answer not in ('OK', 'INCOMPLETE'):
            return out, answer, idx
    raise ValueError("Failed to corrupt")

class SyntaxErrorDetection(Task):
    summary = "Locate syntax errors or grammatical perturbations in generated sentences."
    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig())
        self.config.perturbation_rate = 0.0

    def generate_entry(self):
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
                if not 3 <= len(toks) <= self.config.max_tokens:
                    continue

                roll = random.random()
                if roll < 0.15:
                    ans, idx = first_error(g, toks)
                    if ans != 'OK':
                        continue
                    out = toks
                elif roll < 0.30:
                    out = toks[:random.randint(1, len(toks) - 1)]
                    ans, idx = first_error(g, out)
                    if ans != 'INCOMPLETE':
                        continue
                else:
                    try:
                        out, ans, idx = corrupt_once(g, toks)
                    except ValueError:
                        continue
                if len(out) > self.config.max_tokens:
                    continue

                return Entry(
                    edict(g=grammar_text(g, self.config), start=str(g.start()),
                          tokens=out, error_index=idx),
                    ans
                )
        raise ValueError("Failed to generate locate-error task")

    def render_prompt(self, meta):
        error_format = "Answer OK, INCOMPLETE, or ERROR token for the first invalid token."
        if len(meta.tokens) != len(set(meta.tokens)):
            error_format += (
                " If that token repeats in STRING, append its 1-based occurrence "
                "as @occurrence."
            )
        return (
            f"(START)\n{meta.start}\n\n"
            f"(GRAMMAR)\n{meta.g}\n\n"
            f"(STRING)\n{' '.join(meta.tokens)}\n\n"
            f"{error_format}"
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip() == entry["answer"]) if answer else 0.0


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


def _window_fill_analysis(grammar, prefix, k, suffix=(), max_states=1024,
                          max_fills=None):
    """Return (valid fills, reachable windows), sharing the expensive search."""
    prefix, suffix = list(prefix), list(suffix)
    parser = EarleyChartParser(grammar)
    nullable, first = _compute_nullable_and_first(grammar)

    frontier = {()}
    for _ in range(k):
        nxt = set()
        for win in frontier:
            toks, _ = _exact_next_tokens_and_stop(
                grammar, prefix + list(win), parser, nullable, first
            )
            for tok in toks:
                nxt.add(win + (tok,))
                if len(nxt) > max_states:
                    return [], []
        if not nxt:
            return [], []
        frontier = nxt

    reachable = [list(w) for w in sorted(frontier)]
    fills = []
    if not suffix:
        accepts = lambda w: _exact_next_tokens_and_stop(
            grammar, prefix + w, parser, nullable, first
        )[1]
    else:
        accepts = lambda w: next(parser.parse(prefix + w + suffix), None) is not None
    for window in reachable:
        if accepts(window):
            fills.append(window)
            if max_fills is not None and len(fills) >= max_fills:
                break
    return fills, reachable


def exact_window_fills(grammar, prefix, k, suffix=(), max_states=1024):
    """All distinct k-token windows W such that prefix + W + suffix is grammatical.
    Empty suffix => W must yield STOP (original `exact_completions` behavior).
    Returns [] (safe skip) on state-space overflow."""
    return _window_fill_analysis(grammar, prefix, k, suffix, max_states)[0]


def reachable_windows(grammar, prefix, k, max_states=1024):
    """All length-k token windows reachable after prefix, whether or not they can stop."""
    parser = EarleyChartParser(grammar)
    nullable, first = _compute_nullable_and_first(grammar)
    frontier = {()}
    for _ in range(k):
        nxt = {
            win + (token,)
            for win in frontier
            for token in _exact_next_tokens_and_stop(
                grammar, list(prefix) + list(win), parser, nullable, first,
            )[0]
        }
        if not nxt or len(nxt) > max_states:
            return []
        frontier = nxt
    return [list(window) for window in sorted(frontier)]


class ConstrainedContinuation(Task):
    """Recover a unique fixed-length span inside a grammar-constrained sentence."""
    summary = "Complete a uniquely determined fixed-length span using a formal grammar."

    def __init__(self, config=None):
        super().__init__(config=config or GrammarConfig())
        self.config.prob_resampling_grammar = 0.0  # needed for speed
        self.config.min_k = max(3, self.config.min_k)
        self.balancing_key_ratio = 0.25

    def generate_entry(self):
        for _ in range(200):
            with resampled_grammar(self.config, productive_only=True) as g:
                try:
                    sent = (gramforge_generate(
                        nltk_to_gramforge(g),
                        depth=self.config.max_prod_depth,
                        min_depth=self.config.min_prod_depth,
                        mode=self.config.gramforge_algorithm,
                    ) @ "lang").split()
                except (ValueError, RecursionError):
                    continue
                if not self.config.min_k <= len(sent) <= self.config.max_tokens:
                    continue

                slots = [(start, k)
                         for k in range(self.config.min_k, self.config.max_k + 1)
                         for start in range(0, len(sent) - k + 1)]
                random.shuffle(slots)
                slots.sort(key=lambda x: not (x[0] and x[0] + x[1] < len(sent)))

                for start, k in slots[:16]:
                    prefix = sent[:start]
                    suffix = sent[start + k:]
                    cands, reachable = _window_fill_analysis(
                        g, prefix, k, suffix,
                        max_states=self.config.max_options, max_fills=2,
                    )
                    target = sent[start:start + k]
                    if cands != [target]:
                        continue
                    if not self.config.min_options <= len(reachable) <= self.config.max_options:
                        continue

                    return Entry(
                        edict(
                            g=grammar_text(g, self.config),
                            start=str(g.start()),
                            k=k,
                            prefix=prefix,
                            suffix=suffix,
                            sentence=" ".join([*prefix, "<HOLE>", *suffix]),
                            n_candidates=len(cands),
                            n_reachable=len(reachable),
                            finite_space_exhaustively_checked=True,
                        ),
                        " ".join(target),
                    )

        raise ValueError("Failed to generate constrained continuation after 200 attempts")

    def render_prompt(self, meta):
        return (
            f"Complete <HOLE> according to the grammar.\n\n"
            f"GRAMMAR:\n{meta.g}\n\n"
            f"SENTENCE:\n{meta.sentence}\n\n"
            f"Return only the missing {meta.k} tokens."
        )

    def score_answer(self, answer, entry):
        pred, gold = str(answer).split(), entry["answer"].split()
        return Levenshtein.normalized_similarity(pred, gold) if pred else 0.0


# --- Stress continuation: valid_next(G, prefix) with delayed recursive state ---
#
# One generic constructor, no per-grammar templates. Given a recursive CFG, it
# builds a paired prefix: identical k-token tail, different valid-next sets. The
# shared window certifies the answer is non-local; it depends on carried parse
# state rather than the visible suffix.

INF = float("inf")

@dataclass
class StressContinuationConfig(Config):
    depth: int = 2
    n_types: int = 3
    window: int = 4
    max_answer: int = 8
    def apply_difficulty(self, level):
        self.depth += level
        self.n_types = min(6, self.n_types + level)

def _typed_dyck(n_types):
    pairs = [("(", ")"), ("[", "]"), ("<", ">"), ("{", "}"),
             ("/", "\\"), (":", ";")][:n_types]
    items = " | ".join(f"'{o}' S '{c}'" for o, c in pairs) + " | 'x'"
    return CFG.fromstring(f"S -> Item Tail\nTail -> Item Tail |\nItem -> {items}")

def _stress_sources(config):
    return {
        "dyck": _typed_dyck(config.n_types),
        "agreement": CFG.fromstring("""
            S -> NP_sg VP_sg | NP_pl VP_pl
            NP_sg -> 'the' N_sg RC | 'the' N_sg
            NP_pl -> 'the' N_pl RC | 'the' N_pl
            RC -> 'that' NP_sg V_sg | 'that' NP_pl V_pl
            VP_sg -> 'runs' | 'sleeps'
            VP_pl -> 'run' | 'sleep'
            V_sg -> 'sees' | 'likes'
            V_pl -> 'see' | 'like'
            N_sg -> 'student' | 'teacher'
            N_pl -> 'students' | 'teachers'
        """),
        "filler_gap": CFG.fromstring("""
            S -> WHO AUX NP Vt | WHY AUX NP Adj
            WHO -> 'what' | 'who'
            WHY -> 'why' | 'when'
            AUX -> 'do'
            NP -> 'the' N RC | 'the' N
            RC -> 'that' NP Vt
            N -> 'kids' | 'cooks'
            Vt -> 'see' | 'like' | 'find'
            Adj -> 'happy' | 'sad' | 'kind'
        """),
    }

def _render_rules(g):
    return "\n".join(str(p) for p in g.productions())

def _stress_answer(g, prefix):
    toks, stop, _ = get_valid_next_tokens(g, prefix)
    out = set(toks)
    if stop:
        out.add("STOP")
    return out

def _by_lhs(g):
    d = defaultdict(list)
    for p in g.productions():
        d[p.lhs()].append(p)
    return d

def _reaches(by):
    reach = {nt: {s for p in ps for s in p.rhs()
                  if isinstance(s, Nonterminal)}
             for nt, ps in by.items()}
    changed = True
    while changed:
        changed = False
        for nt in reach:
            add = set().union(*(reach.get(m, set()) for m in reach[nt]))
            add -= reach[nt]
            if add:
                reach[nt] |= add
                changed = True
    return reach

def _min_height(by):
    h = {nt: INF for nt in by}
    changed = True
    while changed:
        changed = False
        for nt, prods in by.items():
            for p in prods:
                v = 1 + max((0 if isinstance(s, str) else h.get(s, INF)
                             for s in p.rhs()), default=0)
                if v < h[nt]:
                    h[nt] = v
                    changed = True
    return h

def _terminable(p, h):
    return all(isinstance(s, str) or h.get(s, INF) < INF for s in p.rhs())

def _children(by, x):
    return {s for p in by[x] for s in p.rhs() if isinstance(s, Nonterminal)}

def _expand_min(sym, h, by):
    if isinstance(sym, str):
        return [sym]
    p = min((p for p in by[sym] if _terminable(p, h)),
            key=lambda p: max((0 if isinstance(s, str) else h[s]
                               for s in p.rhs()), default=0))
    return [t for s in p.rhs() for t in _expand_min(s, h, by)]

def _expand_deep(sym, depth, h, by, reach, rec):
    if isinstance(sym, str) or depth <= 0 or sym not in rec:
        return _expand_min(sym, h, by)
    reentr = lambda s: isinstance(s, Nonterminal) and (
        s == sym or sym in reach.get(s, set()))
    rp = [p for p in by[sym] if _terminable(p, h)
          and any(reentr(s) for s in p.rhs())]
    if not rp:
        return _expand_min(sym, h, by)
    p = random.choice(rp)
    ri = next(i for i, s in enumerate(p.rhs()) if reentr(s))
    return [t for i, s in enumerate(p.rhs())
            for t in (_expand_deep(s, depth - 1, h, by, reach, rec)
                      if i == ri else _expand_min(s, h, by))]

def _route_to(sym, R, shared, h, by, reach, _d=0):
    if sym == R:
        return list(shared)
    if isinstance(sym, str) or _d > 40:
        return _expand_min(sym, h, by) if isinstance(sym, Nonterminal) else [sym]
    reaching = lambda s: isinstance(s, Nonterminal) and (
        s == R or R in reach.get(s, set()))
    cands = [p for p in by[sym] if any(reaching(s) for s in p.rhs())]
    if not cands:
        return _expand_min(sym, h, by)
    p = random.choice(cands)
    ri = next(j for j, s in enumerate(p.rhs()) if reaching(s))
    out = []
    for j, s in enumerate(p.rhs()):
        if j == ri:
            out += _route_to(s, R, shared, h, by, reach, _d + 1)
        else:
            out += _expand_min(s, h, by) if isinstance(s, Nonterminal) else [s]
    return out

def _feature_sites(by, reach, rec):
    sites = []
    for prods in by.values():
        for p, q in ((a, b) for a in prods for b in prods if a is not b):
            for i in range(min(len(p.rhs()), len(q.rhs())) - 1):
                xp, xq = p.rhs()[i], q.rhs()[i]
                if not (isinstance(xp, Nonterminal) and isinstance(xq, Nonterminal)):
                    continue
                common = ({xp} | reach.get(xp, set())) & (
                    {xq} | reach.get(xq, set())) & rec
                if not common:
                    continue
                kids = common & _children(by, xp) & _children(by, xq)
                R = xp if xp == xq and xp in common else min(kids or common, key=str)
                sites.append((p, q, i, R))
    return sites

def _build_side(prod, i, R, shared, h, by, reach):
    out = []
    for j, s in enumerate(prod.rhs()):
        if j < i:
            out += _expand_min(s, h, by) if isinstance(s, Nonterminal) else [s]
        elif j == i:
            out += _route_to(s, R, shared, h, by, reach)
        else:
            break
    return out

def _analyze(g):
    by = _by_lhs(g)
    reach = _reaches(by)
    rec = {nt for nt in reach if nt in reach[nt]}
    return by, reach, rec, _min_height(by), _feature_sites(by, reach, rec)

def _stress_pair(g, analysis, depth, k, max_answer, tries=60):
    by, reach, rec, h, sites = analysis
    for p, q, i, R in random.sample(sites, min(tries, len(sites))):
        shared = _expand_deep(R, depth, h, by, reach, rec)
        pa = _build_side(p, i, R, shared, h, by, reach)
        pb = _build_side(q, i, R, shared, h, by, reach)
        if len(pa) < k or len(pb) < k or pa[-k:] != pb[-k:]:
            continue
        aa, ab = _stress_answer(g, pa), _stress_answer(g, pb)
        if aa == ab or not (aa and ab):
            continue
        if max(len(aa), len(ab)) > max_answer:
            continue
        return (pa, aa), (pb, ab), dict(lhs=str(p.lhs()), R=str(R))
    return None

class StressContinuation(DevTask):
    def __init__(self, config=None):
        super().__init__(config=config or StressContinuationConfig())
        self._sources = _stress_sources(self.config)
        self._analysis = {name: _analyze(g) for name, g in self._sources.items()}
        self.bank = defaultdict(list)

    def generate_entry(self):
        names = list(self._sources)
        k = self.config.window
        for _ in range(200):
            name = random.choice(names)
            g = self._sources[name]
            depth = self.config.depth + k + random.randint(0, 2)
            pair = _stress_pair(g, self._analysis[name], depth, k,
                                self.config.max_answer)
            if not pair:
                continue
            (pa, aa), (pb, ab), m = pair
            for pfx, ans in ((pa, aa), (pb, ab)):
                self.bank[tuple(pfx[-k:])].append(frozenset(ans))
            prefix, answer = random.choice([(pa, aa), (pb, ab)])
            meta = edict(g=_render_rules(g), prefix=prefix, source=name,
                         depth=depth, lhs=m["lhs"], feature=m["R"],
                         answer_size=len(answer), window=" ".join(prefix[-k:]),
                         suffix_flip=True)
            return Entry(meta, "|".join(sorted(answer)))
        raise ValueError("Failed to generate StressContinuation case after 200 attempts")

    def render_prompt(self, meta):
        pfx = " ".join(meta.prefix) if meta.prefix else "<empty>"
        return (
            "Given this projected CFG and prefix, list every terminal that may legally come next.\n"
            "If the prefix is already a complete string, include STOP. "
            "The answer is sorted alphabetically and separated by |.\n"
            f"(GRAMMAR)\n{meta.g}\n\n(PREFIX)\n{pfx}"
        )

    def score_answer(self, answer, entry):
        try:
            ref = {x.strip() for x in entry["answer"].split("|") if x.strip()}
            ans = {x.strip() for x in str(answer).split("|") if x.strip()}
            return len(ref & ans) / max(1, len(ref | ans))
        except Exception:
            return 0.0
