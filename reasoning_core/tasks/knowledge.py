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
_FULL_W2SIDS = defaultdict(set)

_BANNED_WORDS = frozenset({
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'weekday', 'weekend', 'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
})

def _load_wn():
    global _FULL_WORDS, _FULL_W2S, _FULL_S2W, _FULL_W2LEX, _FULL_LEX2W, _FULL_W2SIDS
    if _FULL_WORDS: return
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    lemma_pos_counts = defaultdict(lambda: defaultdict(int))
    for syn in wn.all_synsets():
        pos = syn.pos()
        for lem in syn.lemmas():
            lemma_pos_counts[lem.name().lower()][pos] += lem.count()

    pairs = {}
    for syn in wn.all_synsets('n'):
        for lem in syn.lemmas():
            w = lem.name().lower()
            if not w.isalpha() or w in pairs or w in _BANNED_WORDS: 
                continue
            
            syns = wn.synsets(w, 'n')
            if not syns or syns[0] != syn or len(syns) > 2: 
                continue
            
            counts = lemma_pos_counts[w]
            total = sum(counts.values())
            if total == 0 or (counts['n'] / total) < 0.6:
                continue
                
            z = zipf_frequency(w, 'en')
            if z >= 3.0:
                pairs[w] = (syn, z)

    seen_syns = set()
    for w, (s, _) in sorted(pairs.items(), key=lambda x: -x[1][1]):
        _FULL_W2SIDS[w] = {syn.name() for syn in wn.synsets(w, 'n')}
        if s not in seen_syns:
            _FULL_W2S[w] = s
            _FULL_S2W[s] = w
            
            lex = s.lexname()
            _FULL_W2LEX[w] = lex
            _FULL_LEX2W[lex].append(w)
            
            _FULL_WORDS.append(w)
            seen_syns.add(s)
            
    assert set(_FULL_W2SIDS) >= set(_FULL_WORDS), "Missing synset IDs for validated words"

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
            self._g_odd_one_out,
            lambda: self._g_common_category(2), lambda: self._g_hypernym(3)
        ]

    def _pick(self, n=1):
        pool = _FULL_WORDS[:min(self.config.n_words, len(_FULL_WORDS))]
        return random.sample(pool, n) if n > 1 else random.choice(pool)

    def _s(self, w): return _FULL_W2S.get(w)
    def _w(self, s): return _FULL_S2W.get(s)
    def _ws(self, ss): return {w for s in ss if (w := self._w(s))}

    def _too_abstract(self, s, strict=False):
        if s.min_depth() < 6 or len(s.hyponyms()) > 100: return True
        banned_names = frozenset({
            'entity', 'abstraction', 'physical_entity', 'thing', 
            'object', 'whole', 'person', 'attribute', 'causal_agent', 
            'matter', 'measure', 'communication', 'event', 'act', 'group',
            'state', 'process', 'happening', 'ending', 'instrumentality', 'equipment'
        })
        banned_lex = frozenset({'noun.cognition', 'noun.communication', 'noun.state', 'noun.feeling', 'noun.attribute'})
        
        if strict:
            return s.name().split('.')[0] in banned_names or s.lexname() in banned_lex
        return s.name().split('.')[0] in banned_names

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

    def _get_synonym_block_sids(self, target_s):
        parents = target_s.hypernyms()
        sibs = {h for p in parents for h in p.hyponyms()} - {target_s}
        return {target_s.name()} | {syn.name() for sib in sibs for syn in wn.synsets(self._w(sib) or '', 'n')}

    def _noise(self, exclude, s=None, n=None, gold_sids=None):
        n = n if n is not None else self.config.n_distractors
        gold_sids = set(gold_sids or [])
        ex, out = set(exclude), []
        
        def _valid(word):
            if word in ex: return False
            if not gold_sids: return True
            return not bool(_FULL_W2SIDS.get(word, set()) & gold_sids)

        if s:
            lex = _FULL_W2LEX.get(self._w(s)) or s.lexname()
            same_lex = [w for w in _FULL_LEX2W[lex] if _valid(w)]
            random.shuffle(same_lex)
            out = same_lex[:n]
            ex.update(out)
            
        for _ in range(n * 4):
            if len(out) >= n: break
            r = self._pick()
            if _valid(r):
                out.append(r)
                ex.add(r)
        return out[:n]

    def _sib_noise(self, answer, s=None, n=None, gold_sids=None):
        n = n if n is not None else self.config.n_distractors
        pool = list((self._ws(self._siblings(s)) - {answer}) if s else set())
        random.shuffle(pool)
        
        if gold_sids:
            pool = [p for p in pool if not bool(_FULL_W2SIDS.get(p, set()) & set(gold_sids))]
            
        out = pool[:n]
        return out + self._noise(set(out) | {answer}, s, max(0, n - len(out)), gold_sids)

    def _g_hypernym(self, depth=1):
        w = self._pick(); s = self._s(w)
        anc = self._up(s, depth)
        if not anc or self._too_abstract(anc) or not (a := self._w(anc)): return None
        
        gold_sids = self._get_synonym_block_sids(anc)
        
        expr = w
        for _ in range(depth): expr = f"hypernym({expr})"
        cot = f"{w} is a type of {a}" if depth == 1 else f"{a} is {depth} levels above {w}"
        return expr, a, 'word', self._sib_noise(a, anc, gold_sids=list(gold_sids)), cot, [anc.name()]

    def _g_hyponyms(self):
        w = self._pick(); s = self._s(w)
        if self._too_abstract(s, strict=True): return None
        
        full_kids = list(self._ws(set(s.hyponyms())))
        if len(full_kids) < 2: return None
        
        gold_full_sids = [self._s(k).name() for k in full_kids]
        kids = sorted(random.sample(full_kids, 8) if len(full_kids) > 8 else full_kids)
        sampled_sids = [self._s(k).name() for k in kids]
        
        return f"hyponyms({w})", kids, 'set', self._noise(kids, s, gold_sids=gold_full_sids), f"Types of {w}: {', '.join(kids)}", sampled_sids

    def _g_cohyponyms(self):
        w = self._pick(); s = self._s(w)
        if self._too_abstract(self._up(s), strict=True): return None
        
        full_sibs = list(self._ws(self._siblings(s)) - {w})
        if len(full_sibs) < 2: return None
        
        gold_full_sids = [self._s(x).name() for x in full_sibs]
        sibs = sorted(random.sample(full_sibs, min(len(full_sibs), random.randint(3, 8))))
        sampled_sids = [self._s(x).name() for x in sibs]
        
        return f"cohyponyms({w})", sibs, 'set', self._noise(sibs, s, gold_sids=gold_full_sids), f"{', '.join(sibs)} are in the same category as {w}", sampled_sids

    def _g_is_a(self):
        w = self._pick()
        all_senses = wn.synsets(w, 'n')
        if not all_senses: return None
        
        s = random.choice(all_senses)
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
            return f"is_a({w}, {cat})", 'True', 'bool', [], f"{w} is a type of {cat}", []
        
        for anc_s, _ in chain:
            sibs = [x for x in self._ws(self._siblings(anc_s)) if self._s(x) and not self._too_abstract(self._s(x))]
            random.shuffle(sibs)
            for cat in sibs:
                cat_s = self._s(cat)
                
                valid_negative = True
                for sense in all_senses:
                    lchs = sense.lowest_common_hypernyms(cat_s)
                    lch_depth = max(lchs, key=lambda x: x.min_depth()).min_depth() if lchs else 0
                    
                    if (lch_depth >= 5 or 
                        self._is_ancestor(sense, cat_s) or 
                        self._is_ancestor(cat_s, sense) or 
                        (set(sense.hypernyms()) & set(cat_s.hypernyms()))):
                        valid_negative = False
                        break
                        
                if valid_negative:
                    return f"is_a({w}, {cat})", 'False', 'bool', [], f"{w} is not a type of {cat}", []
        return None

    def _g_lch(self):
        w1, w2 = self._pick(2)
        s1, s2 = self._s(w1), self._s(w2)
        if set(s1.hypernyms()) & set(s2.hypernyms()): return None
        lchs = s1.lowest_common_hypernyms(s2)
        if not lchs: return None
        lch = max(lchs, key=lambda x: x.min_depth())
        if self._too_abstract(lch) or not (a := self._w(lch)) or a in (w1, w2): return None
        
        gold_sids = self._get_synonym_block_sids(lch)
        return f"lowest_common_hypernym({w1}, {w2})", a, 'word', self._sib_noise(a, lch, gold_sids=list(gold_sids)), f"{w1} and {w2} are both types of {a}", [lch.name()]

    def _g_parts(self):
        w = self._pick(); s = self._s(w)
        if s.lexname() in ('noun.location', 'noun.group'): return None
        
        full_parts = sorted(self._ws(set(p for p in s.part_meronyms() if p.lexname().startswith('noun.'))))
        if not full_parts or len(full_parts) > 12: return None
        
        gold_full_sids = [self._s(x).name() for x in full_parts]
        return f"parts_of({w})", full_parts, 'set', self._noise(full_parts, s, gold_sids=gold_full_sids), f"Parts of {w}: {', '.join(full_parts)}", gold_full_sids

    def _g_common_category(self, depth=1):
        w = self._pick(); s = self._s(w)
        cat = self._up(s, depth)
        if not cat or self._too_abstract(cat, strict=True) or not (cat_word := self._w(cat)): return None
        
        members = list(self._ws(set(cat.hyponyms())) - {cat_word})
        if len(members) < 3: return None
        words = sorted(random.sample(members, min(len(members), random.randint(3, 6))))
        
        gold_sids = self._get_synonym_block_sids(cat)
        return f"common_category({', '.join(words)})", cat_word, 'word', self._sib_noise(cat_word, cat, gold_sids=list(gold_sids)), f"{', '.join(words)} are types of {cat_word}", [cat.name()]

    def _g_odd_one_out(self):
        w = self._pick(); s = self._s(w)
        cat = self._up(s)
        if not cat or self._too_abstract(cat, strict=True) or not (cat_word := self._w(cat)): return None
        
        siblings = self._ws(set(cat.hyponyms()))
        group = list(siblings - {w})
        if len(group) < 2: return None
        group = random.sample(group, min(len(group), random.randint(2, 4))) + [w]
        
        cat_descendants = {d.name() for d in cat.closure(lambda x: x.hyponyms())} | {cat.name()}
        
        parent = self._up(cat)
        hard_pool = set()
        if parent:
            for sib in parent.hyponyms():
                if sib != cat:
                    hard_pool.update(self._ws([sib]))
                    hard_pool.update(self._ws(sib.hyponyms()))
                    for niece in sib.hyponyms():
                        hard_pool.update(self._ws(niece.hyponyms()))
                        
        hard_pool = {hw for hw in hard_pool if hw in _FULL_W2SIDS}
        valid_iws = [iw for iw in hard_pool if iw not in siblings and not bool(_FULL_W2SIDS[iw] & cat_descendants)]
        
        if valid_iws:
            iw = random.choice(valid_iws)
        else:
            found = False
            for _ in range(20):
                iw = self._pick()
                if iw not in siblings and not bool(_FULL_W2SIDS.get(iw, set()) & cat_descendants):
                    found = True
                    break
            if not found: return None
            
        words = group + [iw]
        random.shuffle(words)
        return f"odd_one_out({', '.join(words)})", iw, 'word', group, f"{', '.join(sorted(group))} are types of {cat_word}; {iw} is not", [self._s(iw).name()]

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
            
            expr, answer, atype, distractors, cot, gold_sids = r
            
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
                metadata=edict(expr=expr, answer_type=atype, candidates=pool, cot=f"{cot}\n{expr} = {answer_str}", gold_synsets=gold_sids),
                answer=answer_str,
            )
            
        raise RuntimeError(f"Generation failed. Last error: {last_exc}")

    def prompt(self, m):
        cands = ', '.join(m.candidates)
        ctx = "Context: WordNet (relation holds for any valid noun sense)."
        if m.answer_type == 'bool': 
            return f"{ctx}\n\n{m.expr}\nTrue or False?"
        if m.answer_type == 'set':  
            return f"{ctx}\nSelect all {m.expr}\nFrom: [{cands}]\nAnswer is a JSON list."
        return f"{ctx}\n\nSelect {m.expr}\nFrom: [{cands}]\nAnswer is one word."

    def score_answer(self, answer: str, entry) -> float:
        if not answer: return 0.0
        answer = answer.strip()
        gt, atype = entry.answer, entry.metadata.answer_type
        gold_sids = set(entry.metadata.get('gold_synsets', []))

        if atype == 'bool':
            return 1.0 if answer.lower().strip('.') == gt.lower() else 0.0

        if atype == 'set':
            try: pred = set(json.loads(answer))
            except Exception: pred = {x.strip().strip('"\'') for x in answer.strip('[]{}').split(',')} - {''}
            
            if not gold_sids:
                gold = set(json.loads(gt))
                if not gold: return 1.0 if not pred else 0.0
                inter = pred & gold
                if not inter: return 0.0
                p, r = len(inter) / len(pred), len(inter) / len(gold)
                return 2 * p * r / (p + r)

            valid_preds = sum(1 for w in pred if set(s.name() for s in wn.synsets(w.lower(), 'n')) & gold_sids)
            found_gold_sids = set()
            for w in pred:
                found_gold_sids.update(set(s.name() for s in wn.synsets(w.lower(), 'n')) & gold_sids)

            p = valid_preds / max(1, len(pred))
            r = len(found_gold_sids) / max(1, len(gold_sids))
            if p + r == 0: return 0.0
            return 2 * p * r / (p + r)

        if gold_sids:
            pred_sids = {s.name() for s in wn.synsets(answer.lower(), 'n')}
            if pred_sids & gold_sids:
                return 1.0
        return 1.0 if answer.lower() == gt.lower() else 0.0