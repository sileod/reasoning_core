from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config, edict
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
import random, json

# ── Clean word ↔ synset index (1 noun sense only) ─────────────────────────

def _build():
    pairs = {}
    for s in wn.all_synsets('n'):
        w = s.lemmas()[0].name()
        if '_' in w or not w.isalpha() or not w.islower():
            continue
        if len(wn.synsets(w, pos='n')) != 1:
            continue
        z = zipf_frequency(w, 'en')
        if z >= 3.0 and (w not in pairs or z > pairs[w][1]):
            pairs[w] = (s, z)
    ranked = sorted(pairs.items(), key=lambda x: -x[1][1])[:3000]
    w2s = {w: s for w, (s, _) in ranked}
    s2w = {s: w for w, s in w2s.items()}
    return w2s, s2w

_W2S, _S2W = None, None
def _idx():
    global _W2S, _S2W
    if _W2S is None:
        _W2S, _S2W = _build()
    return _W2S, _S2W

def _w(s): return _idx()[1].get(s)
def _s(w): return _idx()[0].get(w)
def _ws(ss): return {_idx()[1][s] for s in ss if s in _idx()[1]}
def _pick(n=1):
    ws = list(_idx()[0])
    return random.sample(ws, n) if n > 1 else random.choice(ws)

# ── Graph ──────────────────────────────────────────────────────────────────

def _up(s, n=1):
    for _ in range(n):
        hs = s.hypernyms()
        if not hs: return None
        s = hs[0]
    return s

def _siblings(s):
    out = set()
    for h in s.hypernyms():
        out |= set(h.hyponyms())
    out.discard(s)
    return out

def _too_abstract(s): return s.min_depth() < 4

def _noise(exclude, n=5):
    """Random vocab words not in exclude set."""
    ex = set(exclude)
    out = []
    for _ in range(n * 4):
        r = _pick()
        if r not in ex:
            out.append(r)
            ex.add(r)
        if len(out) >= n:
            break
    return out

def _sib_noise(answer, s=None, n=4):
    """Sibling-based distractors + random fallback."""
    pool = _ws(_siblings(s)) - {answer} if s else set()
    pool = list(pool)
    random.shuffle(pool)
    out = pool[:n]
    return out + _noise(set(out) | {answer}, n - len(out))

# ── Generators: return (expr, answer_words, type) or None ─────────────────
# answer_words: str for word/bool, list[str] for set

def _g_hypernym(depth=1):
    w, s = _pick(), None
    s = _s(w)
    anc = _up(s, depth)
    if not anc or _too_abstract(anc): return None
    a = _w(anc)
    if not a: return None
    expr = w
    for _ in range(depth):
        expr = f"hypernym({expr})"
    return expr, a, 'word', _sib_noise(a, anc)

def _g_hyponyms():
    w = _pick(); s = _s(w)
    kids = sorted(_ws(set(s.hyponyms())))
    if len(kids) < 2: return None
    if len(kids) > 8: kids = sorted(random.sample(kids, 8))
    return f"hyponyms({w})", kids, 'set', _noise(kids)

def _g_cohyponyms():
    w = _pick(); s = _s(w)
    sibs = sorted(_ws(_siblings(s)) - {w})
    if len(sibs) < 2: return None
    if len(sibs) > 8: sibs = sorted(random.sample(sibs, 8))
    return f"cohyponyms({w})", sibs, 'set', _noise(sibs)

def _g_is_a():
    w = _pick(); s = _s(w)
    chain = []
    cur = s
    for _ in range(6):
        cur = _up(cur)
        if not cur: break
        cw = _w(cur)
        if cw and not _too_abstract(cur):
            chain.append((cur, cw))
    if not chain: return None
    if random.random() < 0.5:
        _, cat = random.choice(chain)
        return f"is_a({w}, {cat})", 'True', 'bool', []
    else:
        for anc_s, _ in chain:
            sibs = [x for x in _ws(_siblings(anc_s)) if _s(x) and not _too_abstract(_s(x))]
            if sibs:
                cat = random.choice(sibs)
                ancestors = set()
                cur = s
                for _ in range(12):
                    cur = _up(cur)
                    if not cur: break
                    ancestors.add(cur)
                if _s(cat) not in ancestors:
                    return f"is_a({w}, {cat})", 'False', 'bool', []
    return None

def _g_lch():
    w1, w2 = _pick(2)
    s1, s2 = _s(w1), _s(w2)
    lchs = s1.lowest_common_hypernyms(s2)
    if not lchs: return None
    lch = lchs[0]
    if _too_abstract(lch): return None
    a = _w(lch)
    if not a or a in (w1, w2): return None
    return f"lowest_common_hypernym({w1}, {w2})", a, 'word', _sib_noise(a, lch)

def _g_parts():
    w = _pick(); s = _s(w)
    parts = sorted(_ws(set(s.part_meronyms())))
    if not parts: return None
    return f"parts_of({w})", parts, 'set', _noise(parts)

def _g_antonym():
    w = _pick()
    for s in wn.synsets(w):
        for lem in s.lemmas():
            if lem.name() == w:
                for ant in lem.antonyms():
                    a = ant.name()
                    if a.isalpha() and a.islower() and '_' not in a:
                        return f"antonym({w})", a, 'word', _sib_noise(a)
    return None

def _g_siblings_composed():
    w = _pick(); s = _s(w)
    p = _up(s)
    if not p or _too_abstract(p): return None
    kids = sorted(_ws(set(p.hyponyms())) - {w})
    if len(kids) < 2: return None
    if len(kids) > 8: kids = sorted(random.sample(kids, 8))
    return f"hyponyms(hypernym({w})) \\ {{{w}}}", kids, 'set', _noise(kids)

def _g_cousins():
    w = _pick(); s = _s(w)
    gp = _up(s, 2)
    if not gp or _too_abstract(gp): return None
    cousins = set()
    for uncle in gp.hyponyms():
        cousins |= _ws(set(uncle.hyponyms()))
    cousins.discard(w)
    cousins = sorted(cousins)
    if len(cousins) < 2: return None
    if len(cousins) > 8: cousins = sorted(random.sample(cousins, 8))
    return f"cousins({w})", cousins, 'set', _noise(cousins)

_GENS = [
    (1, _g_hypernym), (1, _g_hyponyms), (1, _g_parts), (1, _g_antonym),
    (2, lambda: _g_hypernym(2)), (2, _g_cohyponyms), (2, _g_is_a),
    (2, _g_lch), (2, _g_siblings_composed),
    (3, lambda: _g_hypernym(3)), (3, _g_cousins),
]

# ── Task ───────────────────────────────────────────────────────────────────

@dataclass
class LexicalKnowledgeConfig(Config):
    max_complexity: int = 2
    max_retries: int = 80
    def update(self, c=1):
        self.max_complexity = min(self.max_complexity + c, 3)

class LexicalKnowledge(Task):
    def __init__(self, config=LexicalKnowledgeConfig()):
        super().__init__(config=config)
        
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    def generate(self):
        cfg = self.config
        eligible = [(c, g) for c, g in _GENS if c <= cfg.max_complexity]
        for _ in range(cfg.max_retries):
            _, gen = random.choice(eligible)
            try:
                r = gen()
            except Exception:
                continue
            if not r:
                continue
            expr, answer, atype, distractors = r

            # Build candidate pool: answer(s) + distractors, shuffled
            if atype == 'set':
                pool = sorted(set(answer) | set(distractors))
                random.shuffle(pool)
                answer_str = json.dumps(sorted(answer))
            elif atype == 'bool':
                pool = ['True', 'False']
                random.shuffle(pool)
                answer_str = str(answer)
            else:  # word
                pool = sorted({answer} | set(distractors))
                random.shuffle(pool)
                answer_str = answer

            return Problem(
                metadata=edict(
                    expr=expr, answer_type=atype,
                    candidates=pool,
                    cot=f"{expr} = {answer_str}",
                ),
                answer=answer_str,
            )
        raise RuntimeError("generation failed")

    def prompt(self, m):
        cands = ', '.join(m.candidates)
        if m.answer_type == 'bool':
            return f"{m.expr}\n\nTrue or False?"
        if m.answer_type == 'set':
            return (
                f"{m.expr}\n\n"
                f"From: [{cands}]\n"
                f"Select all that apply as a JSON list."
            )
        return (
            f"{m.expr}\n\n"
            f"From: [{cands}]\n"
            f"Answer with one word."
        )

    def score_answer(self, answer: str, entry) -> float:
        if not answer: return 0.0
        answer = answer.strip()
        gt, atype = entry.answer, entry.metadata.answer_type
        if atype == 'bool':
            return 1.0 if answer.lower().strip('.') == gt.lower() else 0.0
        if atype == 'set':
            try:
                pred = set(json.loads(answer))
            except Exception:
                pred = {x.strip().strip('"\'') for x in answer.strip('[]{}').split(',')} - {''}
            gold = set(json.loads(gt))
            if not gold: return 1.0 if not pred else 0.0
            inter = pred & gold
            if not inter: return 0.0
            p, r = len(inter) / len(pred), len(inter) / len(gold)
            return 2 * p * r / (p + r)
        return 1.0 if answer.lower() == gt.lower() else 0.0
