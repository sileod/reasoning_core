from dataclasses import dataclass
from collections import defaultdict
from reasoning_core.template import Task, Problem, Config, edict
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
import random
import json

_FULL_WORDS = []
_FULL_W2S = {}
_FULL_S2W = {}
_FULL_W2LEX = {}
_FULL_LEX2W = defaultdict(list)

def _load_wn():
    global _FULL_WORDS, _FULL_W2S, _FULL_S2W, _FULL_W2LEX, _FULL_LEX2W
    if _FULL_WORDS: return
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    pairs = {}
    for syn in wn.all_synsets('n'):
        for lem in syn.lemmas():
            w = lem.name().lower()
            if not w.isalpha() or w in pairs: 
                continue
            
            syns = wn.synsets(w, 'n')
            if not syns or syns[0] != syn or len(syns) > 2: 
                continue
            
            total_n_count = sum(l.count() for s in syns for l in s.lemmas() if l.name().lower() == w)
            if total_n_count == 0 or (lem.count() / total_n_count) < 0.5:
                continue
                
            z = zipf_frequency(w, 'en')
            if z >= 3.0:
                pairs[w] = (syn, z)

    seen_syns = set()
    for w, (s, _) in sorted(pairs.items(), key=lambda x: -x[1][1]):
        if s not in seen_syns:
            _FULL_W2S[w] = s
            _FULL_S2W[s] = w
            
            lex = s.lexname()
            _FULL_W2LEX[w] = lex
            _FULL_LEX2W[lex].append(w)
            
            _FULL_WORDS.append(w)
            seen_syns.add(s)

@dataclass
class LexicalKnowledgeConfig(Config):
    n_words: int = 600
    max_retries: int = 80
    n_distractors: int = 5
    
    def update(self, c=1):
        self.n_words = min(int(self.n_words * (1 + c)), len(_FULL_WORDS) or float('inf'))

class LexicalKnowledge(Task):
    def __init__(self, config=LexicalKnowledgeConfig()):
        super().__init__(config=config)
        _load_wn()
        self.generators = [
            lambda: self._g_hypernym(1), self._g_hyponyms, self._g_parts,
            lambda: self._g_common_category(1), lambda: self._g_hypernym(2),
            self._g_cohyponyms, self._g_is_a, self._g_lch, 
            self._g_siblings_composed, self._g_odd_one_out,
            lambda: self._g_common_category(2), lambda: self._g_hypernym(3),
            self._g_cousins,
        ]

    def _pick(self, n=1):
        pool = _FULL_WORDS[:min(self.config.n_words, len(_FULL_WORDS))]
        return random.sample(pool, n) if n > 1 else random.choice(pool)

    def _s(self, w): return _FULL_W2S.get(w)
    def _w(self, s): return _FULL_S2W.get(s)
    def _ws(self, ss): return {w for s in ss if (w := self._w(s))}

    def _too_abstract(self, s):
        if s.min_depth() < 6 or len(s.hyponyms()) > 100: return True
        return s.name().split('.')[0] in {
            'entity', 'abstraction', 'physical_entity', 'thing', 
            'object', 'whole', 'person', 'attribute', 'causal_agent', 
            'matter', 'measure', 'communication', 'event', 'act', 'group',
            'state', 'process', 'happening', 'ending', 'instrumentality', 'equipment'
        }

    def _best_parent(self, s):
        hs = [h for h in s.hypernyms() if not self._too_abstract(h)]
        if not hs: hs = s.hypernyms() 
        if not hs: return None
        return min(hs, key=lambda h: (len(h.hyponyms()), -h.min_depth()))

    def _up(self, s, n=1):
        for _ in range(n):
            s = self._best_parent(s)
            if not s: return None
        return s

    def _siblings(self, s):
        h = self._best_parent(s)
        return set(h.hyponyms()) - {s} if h else set()

    def _is_ancestor(self, child, parent):
        return parent in set(child.closure(lambda x: x.hypernyms()))

    def _noise(self, exclude, s=None, n=None):
        n = n if n is not None else self.config.n_distractors
        ex, out = set(exclude), []
        
        if s:
            lex = _FULL_W2LEX.get(self._w(s)) or s.lexname()
            same_lex = [w for w in _FULL_LEX2W[lex] if w not in ex]
            random.shuffle(same_lex)
            out = same_lex[:n]
            ex.update(out)
            
        for _ in range(n * 4):
            if len(out) >= n: break
            r = self._pick()
            if r not in ex:
                out.append(r)
                ex.add(r)
        return out[:n]

    def _sib_noise(self, answer, s=None, n=None):
        n = n if n is not None else self.config.n_distractors
        pool = list((self._ws(self._siblings(s)) - {answer}) if s else set())
        random.shuffle(pool)
        out = pool[:n]
        return out + self._noise(set(out) | {answer}, s, max(0, n - len(out)))

    def _g_hypernym(self, depth=1):
        w = self._pick(); s = self._s(w)
        anc = self._up(s, depth)
        if not anc or self._too_abstract(anc) or not (a := self._w(anc)): return None
        expr = w
        for _ in range(depth): expr = f"hypernym({expr})"
        cot = f"{w} is a type of {a}" if depth == 1 else f"{a} is {depth} levels above {w}"
        return expr, a, 'word', self._sib_noise(a, anc), cot

    def _g_hyponyms(self):
        w = self._pick(); s = self._s(w)
        if self._too_abstract(s): return None
        kids = list(self._ws(set(s.hyponyms())))
        if len(kids) < 2: return None
        kids = sorted(random.sample(kids, 8) if len(kids) > 8 else kids)
        return f"hyponyms({w})", kids, 'set', self._noise(kids, s), f"Types of {w}: {', '.join(kids)}"

    def _g_cohyponyms(self):
        w = self._pick(); s = self._s(w)
        sibs = list(self._ws(self._siblings(s)) - {w})
        if len(sibs) < 2: return None
        sibs = sorted(random.sample(sibs, min(len(sibs), random.randint(3, 8))))
        return f"cohyponyms({w})", sibs, 'set', self._noise(sibs, s), f"{', '.join(sibs)} are in the same category as {w}"

    def _g_is_a(self):
        w = self._pick(); s = self._s(w)
        cur, chain = s, []
        
        for _ in range(6):
            cur = self._up(cur)
            if not cur: break
            cw = self._w(cur)
            if cw and not self._too_abstract(cur):
                chain.append((cur, cw))
                
        if not chain: return None
        
        if random.random() < 0.5:
            _, cat = random.choice(chain)
            return f"is_a({w}, {cat})", 'True', 'bool', [], f"{w} is a type of {cat}"
        
        for anc_s, _ in chain:
            sibs = [x for x in self._ws(self._siblings(anc_s)) if self._s(x) and not self._too_abstract(self._s(x))]
            random.shuffle(sibs)
            for cat in sibs:
                cat_s = self._s(cat)
                if not self._is_ancestor(s, cat_s) and not self._is_ancestor(cat_s, s) and not (set(s.hypernyms()) & set(cat_s.hypernyms())):
                    return f"is_a({w}, {cat})", 'False', 'bool', [], f"{w} is not a type of {cat}"
        return None

    def _g_lch(self):
        w1, w2 = self._pick(2)
        s1, s2 = self._s(w1), self._s(w2)
        if set(s1.hypernyms()) & set(s2.hypernyms()): return None
        lchs = s1.lowest_common_hypernyms(s2)
        if not lchs: return None
        lch = max(lchs, key=lambda x: x.min_depth())
        if self._too_abstract(lch) or not (a := self._w(lch)) or a in (w1, w2): return None
        return f"lowest_common_hypernym({w1}, {w2})", a, 'word', self._sib_noise(a, lch), f"{w1} and {w2} are both types of {a}"

    def _g_parts(self):
        w = self._pick(); s = self._s(w)
        if s.lexname() in ('noun.location', 'noun.group'): return None
        parts = sorted(self._ws(set(p for p in s.part_meronyms() if p.lexname().startswith('noun.'))))
        if not parts or len(parts) > 12: return None
        return f"parts_of({w})", parts, 'set', self._noise(parts, s), f"Parts of {w}: {', '.join(parts)}"

    def _g_siblings_composed(self):
        w = self._pick(); s = self._s(w)
        p = self._up(s)
        if not p or self._too_abstract(p): return None
        kids = list(self._ws(set(p.hyponyms())) - {w})
        if len(kids) < 2: return None
        kids = sorted(random.sample(kids, min(len(kids), random.randint(3, 8))))
        pw = self._w(p)
        cot = f"{w} and {', '.join(kids)} are types of {pw}" if pw else f"{', '.join(kids)} are siblings of {w}"
        return f"hyponyms(hypernym({w})) \\ {{{w}}}", kids, 'set', self._noise(kids, s), cot

    def _g_cousins(self):
        w = self._pick(); s = self._s(w)
        p = self._up(s)
        gp = self._up(p) if p else None
        if not gp or self._too_abstract(gp) or len(gp.hyponyms()) > 20: return None
        
        cousins = set()
        for uncle in gp.hyponyms():
            if uncle != p and not self._too_abstract(uncle):
                cousins |= self._ws(set(uncle.hyponyms()))
        
        cousins -= self._ws(set(p.hyponyms()))
        cousins.discard(w)
        if len(cousins) < 2 or len(cousins) > 12: return None
        cousins = sorted(random.sample(list(cousins), min(len(cousins), random.randint(3, 8))))
        
        gpw = self._w(gp)
        cot = f"{', '.join(cousins)} share a grandparent ({gpw}) with {w}" if gpw else f"Share grandparent with {w}"
        return f"cousins({w})", cousins, 'set', self._noise(cousins, s), cot

    def _g_common_category(self, depth=1):
        w = self._pick(); s = self._s(w)
        cat = self._up(s, depth)
        if not cat or self._too_abstract(cat) or not (cat_word := self._w(cat)): return None
        members = list(self._ws(set(cat.hyponyms())) - {cat_word})
        if len(members) < 3: return None
        words = sorted(random.sample(members, min(len(members), random.randint(3, 6))))
        return f"common_category({', '.join(words)})", cat_word, 'word', self._sib_noise(cat_word, cat), f"{', '.join(words)} are types of {cat_word}"

    def _g_odd_one_out(self):
        w = self._pick(); s = self._s(w)
        cat = self._up(s)
        if not cat or self._too_abstract(cat) or not (cat_word := self._w(cat)): return None
        
        siblings = self._ws(set(cat.hyponyms()))
        group = list(siblings - {w})
        if len(group) < 2: return None
        group = random.sample(group, min(len(group), random.randint(2, 4))) + [w]
        
        for _ in range(20):
            iw = self._pick()
            if iw in siblings or self._is_ancestor(self._s(iw), cat): continue
            words = group + [iw]
            random.shuffle(words)
            return f"odd_one_out({', '.join(words)})", iw, 'word', group, f"{', '.join(sorted(group))} are types of {cat_word}; {iw} is not"
        return None

    def generate(self):
        cfg = self.config
        last_exc = None
        for _ in range(cfg.max_retries):
            try:
                r = random.choice(self.generators)()
            except Exception as e:
                last_exc = e
                continue
            if not r: continue
            
            expr, answer, atype, distractors, cot = r
            
            if atype == 'set':
                pool = sorted(set(answer) | set(distractors))
                answer_str = json.dumps(sorted(answer))
            elif atype == 'bool':
                pool = ['True', 'False']
                answer_str = str(answer)
            else:
                pool = sorted({answer} | set(distractors))
                answer_str = answer
            
            random.shuffle(pool)
            return Problem(
                metadata=edict(expr=expr, answer_type=atype, candidates=pool, cot=f"{cot}\n{expr} = {answer_str}"),
                answer=answer_str,
            )
            
        raise RuntimeError(f"Generation failed. Last error: {last_exc}")

    def prompt(self, m):
        cands = ', '.join(m.candidates)
        if m.answer_type == 'bool': return f"{m.expr}\n\nTrue or False?"
        if m.answer_type == 'set':  return f"{m.expr}\n\nFrom: [{cands}]\nSelect all that apply as a JSON list."
        return f"{m.expr}\n\nFrom: [{cands}]\nAnswer with one word."

    def score_answer(self, answer: str, entry) -> float:
        if not answer: return 0.0
        answer = answer.strip()
        gt, atype = entry.answer, entry.metadata.answer_type
        if atype == 'bool':
            return 1.0 if answer.lower().strip('.') == gt.lower() else 0.0
        if atype == 'set':
            try: pred = set(json.loads(answer))
            except Exception: pred = {x.strip().strip('"\'') for x in answer.strip('[]{}').split(',')} - {''}
            gold = set(json.loads(gt))
            if not gold: return 1.0 if not pred else 0.0
            inter = pred & gold
            if not inter: return 0.0
            p, r = len(inter) / len(pred), len(inter) / len(gold)
            return 2 * p * r / (p + r)
        return 1.0 if answer.lower() == gt.lower() else 0.0