from gramforge import init_grammar, generate, generate_with_choices
from tqdm.auto import tqdm
from functools import cache
from nltk.parse.generate import generate as nltk_generate
from nltk import CFG, ChartParser
from nltk.parse.earleychart import EarleyChartParser
import sys
from reasoning_core.template import Task, Problem, Config
import random
from pathlib import Path
from nltk.data import path as nltk_path
import string
from easydict import EasyDict as edict
from faker import Faker
import re
from nltk.tree import Tree
from collections import defaultdict
from gramforge.grammars import simple_english_grammar, arith_grammar, dyck_grammar
from gramforge import gramforge_to_nltk
from rapidfuzz.distance import Levenshtein
from itertools import islice
from nltk.grammar import CFG, Nonterminal
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



def sample_cfg(config=GrammarConfig):
    if random.random()>config.random_grammar_prob:
        g = random.choice(existing_grammars)
        # Only trim if grammar is larger than target
        if len(g.productions()) > config.target_num_rules:
            return trim_grammar(g, config.target_num_rules)
        return g
        
    for _ in range(1000):
        MG = meta_grammar(config).start()
        for _ in range(100): 
            x=generate(MG,depth=config.max_depth,min_depth=config.min_depth)
            g = CFG.fromstring(x@"cfg")
            try:
                prods=list(islice(nltk_generate(g ,depth=config.max_prod_depth), 10))
            except (RecursionError, ValueError):
                continue
            if len(prods)>3:
                return g

def perturb(tokens, config=GrammarConfig):
    return random.choice([
        lambda t: random.sample(t, len(t)),
        lambda t: (lambda i: t[:i]+t[i+1:])(random.randrange(len(t))) if len(t)>1 else t,
        #lambda _: (generate(nltk_to_unigram(sample_cfg(config)).get_rules('s', shuffle=True)[0], depth=5) @ 'lang').split()
        lambda _: (generate(nltk_to_gramforge(sample_cfg(config)), depth=5) @ 'lang').split()

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
        g = sample_cfg(config)
        g_u = nltk_to_gramforge(g)
        
        try:
            tokens = (generate(g_u, depth=config.max_prod_depth, min_depth=config.min_prod_depth) @ "lang").split()
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
            "Return the fully parenthesized parse tree of STRING in Lisp style.\n"
            f"{ex}")


    def score_answer(self, answer, entry):
        norm = lambda s: re.sub(r'\s+', ' ', str(s).strip()).replace('"','').replace("'",'')

        reference = entry['answer']
        if not answer: return 0.0
        
        return Levenshtein.normalized_similarity(norm(answer), norm(reference))


def get_valid_next_tokens(grammar, prefix):
    """
    Given a CFG and a prefix (list of tokens), return:
    - set of valid next terminals
    - whether STOP is valid (prefix is a complete sentence)
    - dict mapping each token to its justification from the chart edge
    
    Uses EarleyChartParser to consider ALL possible parse interpretations.
    """
    from functools import lru_cache
    
    parser = EarleyChartParser(grammar)
    
    @lru_cache(maxsize=None)
    def first_with_path(symbol, depth=0):
        """Return dict mapping terminals to derivation paths from symbol (max 2 levels)"""
        if depth > 2:
            return {}
        if isinstance(symbol, str):
            return {symbol: symbol}
        result = {}
        for prod in grammar.productions(lhs=symbol):
            if not prod.rhs():
                continue
            first_sym = prod.rhs()[0]
            if isinstance(first_sym, str):
                result[first_sym] = f"{symbol}→{first_sym}"
            else:
                for tok, path in first_with_path(first_sym, depth+1).items():
                    if tok not in result:
                        # Show one level of derivation instead of →..→
                        result[tok] = f"{symbol}→{first_sym}→{tok}"
        return result
    
    chart = parser.chart_parse(prefix)
    
    valid_tokens = set()
    justifications = {}
    can_stop = False
    n = len(prefix)
    # Use chart.select for efficiency - only look at boundary edges
    for edge in chart.select(end=n):
        if edge.is_complete():
            if edge.start() == 0 and edge.lhs() == grammar.start():
                can_stop = True
                justifications['STOP'] = f"{edge.lhs()}•"
        else:
            nextsym = edge.nextsym()
            if nextsym:
                # Format edge as "A→α•β" style
                lhs = edge.lhs()
                rhs = edge.rhs()
                dot_pos = edge.dot()
                before = ' '.join(str(s) for s in rhs[:dot_pos])
                after = ' '.join(str(s) for s in rhs[dot_pos:])
                edge_str = f"{lhs}→{before}•{after}" if before else f"{lhs}→•{after}"
                
                if isinstance(nextsym, str):
                    valid_tokens.add(nextsym)
                    if nextsym not in justifications:
                        justifications[nextsym] = edge_str
                else:
                    for tok, path in first_with_path(nextsym).items():
                        valid_tokens.add(tok)
                        if tok not in justifications:
                            justifications[tok] = f"{edge_str}, {path}"
    
    return valid_tokens, can_stop, justifications


def _build_cot(tokens, can_stop, justifications):
    """Build CoT string, grouping tokens that share the same edge."""
    parts = []
    
    # Handle STOP first
    if can_stop and 'STOP' in justifications:
        parts.append(f"{justifications['STOP']}⇒STOP")
    
    # Group tokens by their edge (everything before the final →tok)
    edge_to_tokens = defaultdict(list)
    for tok in sorted(tokens):
        if tok in justifications:
            j = justifications[tok]
            # Extract the edge part (before the last →tok)
            edge_key = j.rsplit('→', 1)[0] if '→' in j else j
            edge_to_tokens[edge_key].append(tok)
    
    # Build parts: group if >3 tokens share same edge, else individual
    for edge, toks in sorted(edge_to_tokens.items()):
        if len(toks) > 3:
            parts.append(f"{edge}→{{{','.join(toks)}}}")
        else:
            parts.extend(f"{justifications[t]}⇒{t}" for t in toks)
    
    return "\n".join(parts) if parts else "continuation"


class Continuation(Task):
    """Grammar continuation task using proper CFG parsing."""
    
    def __init__(self, config: GrammarConfig = GrammarConfig()):
        super().__init__(config=config)
        self.balancing_key_ratio = 0.1
        
    def generate(self):
        for _ in range(100):
            g = sample_cfg(self.config)
            
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
                f"Answer sorted alphabetically separated by |, with STOP at the end if complete.\n"
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

def first_error_marked(g, tokens):
    lines = []
    for i, tok in enumerate(tokens):
        valid, _, _ = get_valid_next_tokens(g, tokens[:i])
        if tok not in valid:
            lines.append(f"{tok} ∉ {{{','.join(sorted(valid)[:8])}}}")
            ans = min_context(tokens, i)
            lines.append(f"Answer: {ans}")
            return ans, i, '\n'.join(lines)
        lines.append(f"{tok} ✓")
    _, can_stop, _ = get_valid_next_tokens(g, tokens)
    tag = 'OK' if can_stop else 'INCOMPLETE'
    return tag, -1, '\n'.join(lines)

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
            g = sample_cfg(self.config)
            if len(grammar_terminals(g)) < 2:
                continue
            try:
                toks = (generate(
                    nltk_to_gramforge(g),
                    depth=self.config.max_prod_depth,
                    min_depth=self.config.min_prod_depth
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
            f"Return the shortest contiguous span from STRING that ends at the first invalid token "
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
            return max(0.9, efficiency)


# --- Constrained Generation ---

def exact_completions(grammar, prefix, k, max_states=2048):
    """
    Return all distinct suffixes of exact length k such that prefix+suffix
    is a complete sentence under grammar. Sorted lexicographically.
    """
    prefix = tuple(prefix)
    frontier = {()}

    for _ in range(k):
        nxt = set()
        for suf in frontier:
            toks, _, _ = get_valid_next_tokens(grammar, list(prefix + suf))
            for tok in toks:
                nxt.add(suf + (tok,))
        if not nxt or len(nxt) > max_states:
            return []
        frontier = nxt

    return [
        suf for suf in sorted(frontier)
        if get_valid_next_tokens(grammar, list(prefix + suf))[1]
    ]


def minimal_separating_hints(candidates, target, max_hint_ratio=0.5):
    """
    Exact minimum-cardinality positional hints {i: tok} that isolate `target`
    among `candidates`, subject to a hint budget.

    Returns:
      ({i: tok, ...}, [target]) if successful
      (None, candidates)        if no solution fits the budget
    """
    cands = [tuple(c) for c in candidates]
    target = tuple(target)
    k = len(target)
    max_hints = min(k, int(k * max_hint_ratio + 1e-9))  # floor

    for r in range(1, max_hints + 1):
        for idxs in combinations(range(k), r):
            alive = [c for c in cands if all(c[i] == target[i] for i in idxs)]
            if len(alive) == 1:
                return {i: target[i] for i in idxs}, alive

    return None, cands


def pick_target_with_hints(candidates, max_hint_ratio=0.5):
    """
    Choose a target that can be uniquely identified within the hint budget.
    Prefer targets requiring the fewest hints; break ties randomly.
    """
    feasible = []
    for target in map(tuple, candidates):
        hints, alive = minimal_separating_hints(candidates, target, max_hint_ratio)
        if hints is not None and len(alive) == 1:
            feasible.append((len(hints), target, hints))

    if not feasible:
        return None

    best_n = min(n for n, _, _ in feasible)
    _, target, hints = random.choice([x for x in feasible if x[0] == best_n])
    return list(target), hints


def _format_hints(hints):
    return "none" if not hints else " ".join(f"{i}:{tok}" for i, tok in sorted(hints.items()))


def _hint_cot(prefix, candidates, hints):
    alive = list(candidates)
    lines = [f"{len(alive)} candidates"]
    for i, tok in sorted(hints.items()):
        alive = [c for c in alive if c[i] == tok]
        lines.append(f"hint {i}:{tok} -> {len(alive)} candidate(s)")
    for j, c in enumerate(alive[:8], 1):
        lines.append(f"{j}. {' '.join(prefix + list(c))}")
    return "\n".join(lines)


class ConstrainedContinuation(Task):

    def __init__(
        self,
        config: GrammarConfig = GrammarConfig(),
        min_k=3,
        max_k=4,
        max_options=20,
        max_hint_ratio=0.5,
    ):
        super().__init__(config=config)
        self.min_k = max(3, min_k)   # enforce at least 3 continuation tokens
        self.max_k = max_k
        self.max_options = max_options
        self.max_hint_ratio = max_hint_ratio
        self.balancing_key_ratio = 0.1

    def generate(self):
        for _ in range(200):
            g = sample_cfg(self.config)
            try:
                sentences = [
                    list(s)
                    for s in islice(nltk_generate(g, depth=self.config.max_depth), 80)
                    if len(s) >= self.min_k + 1
                ]
            except (RecursionError, ValueError):
                continue
            if not sentences:
                continue

            sent = random.choice(sentences)
            max_prefix_len = min(5, len(sent) - self.min_k)
            if max_prefix_len < 1:
                continue

            for plen in random.sample(range(1, max_prefix_len + 1), max_prefix_len):
                prefix = sent[:plen]
                ks = list(range(self.min_k, min(self.max_k, len(sent) - plen) + 1))
                random.shuffle(ks)

                for k in ks:
                    cands = exact_completions(g, prefix, k)
                    if not (2 <= len(cands) <= self.max_options):
                        continue

                    picked = pick_target_with_hints(cands, self.max_hint_ratio)
                    if not picked:
                        continue

                    target, hints = picked
                    n_hints = len(hints)
                    if n_hints / k > self.max_hint_ratio:
                        continue

                    return Problem(
                        edict(
                            g="\n".join(str(p) for p in g.productions()),
                            k=k,
                            prefix=prefix,
                            hints={str(i): tok for i, tok in hints.items()},
                            hint_str=_format_hints(hints),
                            n_hints=n_hints,
                            hint_ratio=n_hints / k,
                            n_options=len(cands),
                            cot=_hint_cot(prefix, cands, hints),
                        ),
                        " ".join(target),
                    )

        raise ValueError("Failed to generate constrained continuation")

    def prompt(self, meta):
        pfx = " ".join(meta.prefix) if meta.prefix else "<empty>"
        return (
            f"(GRAMMAR)\n{meta.g}\n\n"
            f"(PREFIX)\n{pfx}\n\n"
            f"Continue PREFIX with exactly {meta.k} tokens to form a complete sentence.\n"
            f"Positional hints (0-indexed within your continuation):\n"
            f"{meta.hint_str}\n\n"
            f"Return only the {meta.k} continuation tokens, space-separated."
        )

    def score_answer(self, answer, entry):
        if not answer:
            return 0.0

        ans, ref = answer.strip().split(), entry["answer"].split()
        if ans == ref:
            return 1.0
        if len(ans) != len(ref):
            return 0.0

        hints = entry.metadata["hints"]
        for i, tok in hints.items():
            i = int(i)  # JSON safety
            if i >= len(ans) or ans[i] != tok:
                return 0.0

        try:
            g = CFG.fromstring(entry.metadata["g"])
            _, can_stop, _ = get_valid_next_tokens(g, list(entry.metadata["prefix"]) + ans)
        except Exception:
            can_stop = False

        matches = sum(a == b for a, b in zip(ans, ref)) / len(ref)
        return (0.4 + 0.4 * matches) if can_stop else 0.2 * matches