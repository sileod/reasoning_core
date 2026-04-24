import random
from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config, edict

ROLES = ['doctor', 'lawyer', 'teacher', 'engineer', 'pilot', 'chef',
         'writer', 'nurse', 'banker', 'farmer', 'scientist', 'baker']
ATTR_GROUPS = [['tall', 'short'], ['young', 'old'],
               ['quiet', 'loud'], ['kind', 'stern']]
VERBS = ['met', 'called', 'praised', 'avoided', 'questioned', 'greeted',
         'thanked', 'watched', 'helped']
NAMES_M = ['John', 'Paul', 'Mark', 'Leo', 'Tom', 'Sam', 'Max', 'Ben']
NAMES_F = ['Mary', 'Anna', 'Jane', 'Eve', 'Sara', 'Lucy', 'Zoe', 'Rita']
PRON = {'m': ('He', 'him'), 'f': ('She', 'her')}
INTRO, NAME, DESC, PRONOUN = 'intro', 'name', 'desc', 'pron'


@dataclass(frozen=True)
class _Entity:
    eid: int; name: str; gender: str; role: str; attrs: tuple


def _pool(n):
    n = min(n, len(NAMES_M) + len(NAMES_F))
    lo, hi = max(1, n - len(NAMES_F)), min(n - 1, len(NAMES_M))
    n_m = random.randint(lo, hi) if lo <= hi else lo
    names = random.sample(NAMES_M, n_m) + random.sample(NAMES_F, n - n_m)
    genders = ['m'] * n_m + ['f'] * (n - n_m)
    pairs = list(zip(names, genders)); random.shuffle(pairs)
    return [_Entity(i, pairs[i][0], pairs[i][1], random.choice(ROLES),
                    tuple(sorted(random.choice(g)
                                 for g in random.sample(ATTR_GROUPS, 2))))
            for i in range(n)]


def _indef(e):
    desc = f"{' '.join(e.attrs)} {e.role}"
    det = "an" if desc[0].lower() in 'aeiou' else "a"
    return f"{det} {desc} named {e.name}"


def _desc(e, pool):
    """Minimal definite NP uniquely picking e out of pool, else None."""
    same = [x for x in pool if x.role == e.role]
    if len(same) == 1:
        return f"the {e.role}"
    for a in e.attrs:
        if sum(a in x.attrs for x in same) == 1:
            return f"the {a} {e.role}"
    if sum(set(e.attrs) <= set(x.attrs) for x in same) == 1:
        return f"the {' '.join(e.attrs)} {e.role}"
    return None


def _pron_ok(e, prev_ents, cur_subj, pos):
    if pos == 'object' and e == cur_subj:       # Principle B
        return False
    if e not in prev_ents:                      # must be in previous sentence
        return False
    ctx = prev_ents | ({cur_subj} if cur_subj else set())
    return not any(x.gender == e.gender and x != e for x in ctx)


def _plan(pool, target, chain_len, n_distractors):
    others = [e for e in pool if e != target]
    out = []
    for _ in range(chain_len):
        o = random.choice(others)
        out.append((target, o) if random.random() < 0.5 else (o, target))
    for _ in range(n_distractors):
        s, o = random.sample(others, 2)
        out.append((s, o))
    random.shuffle(out)
    return out


def _pick(e, pool, introduced, prev_ents, cur_subj, pos, p_pron, p_desc):
    if e not in introduced:
        return _indef(e), INTRO
    prefs = []
    if random.random() < p_pron: prefs.append(PRONOUN)
    if random.random() < p_desc: prefs.append(DESC)
    prefs.append(NAME)
    for m in prefs:
        if m == PRONOUN and _pron_ok(e, prev_ents, cur_subj, pos):
            return PRON[e.gender][0 if pos == 'subject' else 1], PRONOUN
        if m == DESC:
            d = _desc(e, pool)
            if d: return d, DESC
        if m == NAME:
            return e.name, NAME
    return e.name, NAME


def _emit(plan, pool, p_pron, p_desc):
    introduced, lines, mentions = set(), [], []
    for i, (s_e, o_e) in enumerate(plan):
        prev = {m['entity'] for m in mentions if m['sent'] == i - 1}
        v = random.choice(VERBS)
        s_ref, s_mode = _pick(s_e, pool, introduced, prev, None,
                              'subject', p_pron, p_desc)
        introduced.add(s_e)
        o_ref, o_mode = _pick(o_e, pool, introduced, prev, s_e,
                              'object', p_pron, p_desc)
        introduced.add(o_e)
        cap = lambda s: s[:1].upper() + s[1:]
        lines.append(f"({i+1}) {cap(s_ref)} {v} {o_ref}.")
        mentions.extend([
            {'sent': i, 'pos': 'subject', 'surface': s_ref, 'mode': s_mode, 'entity': s_e},
            {'sent': i, 'pos': 'object',  'surface': o_ref, 'mode': o_mode, 'entity': o_e},
        ])
    return lines, mentions


@dataclass
class CoreferenceConfig(Config):
    n_entities: int = 3
    chain_len: int = 2
    n_distractors: int = 2
    p_pronoun: float = 0.7
    p_desc: float = 0.5
    p_shortcut: float = 0.15    # prob. of using a shorter chain for diversity

    def update(self, c=1):
        self.n_entities += c
        self.chain_len += c
        self.n_distractors += c


class Coreference(Task):
    def __init__(self, config=CoreferenceConfig()):
        super().__init__(config=config)

    def generate(self):
        cfg = self.config
        fallback = None
        for _ in range(100):
            clen = cfg.chain_len
            if clen > 2 and random.random() < cfg.p_shortcut:
                clen = random.randint(2, clen - 1)
            pool = _pool(cfg.n_entities)
            target = random.choice(pool)
            plan = _plan(pool, target, clen, cfg.n_distractors)
            lines, mentions = _emit(plan, pool, cfg.p_pronoun, cfg.p_desc)
            cands = [m for m in mentions
                     if m['entity'] == target and m['mode'] in (PRONOUN, DESC)]
            if not cands: continue
            prons = [m for m in cands if m['mode'] == PRONOUN]
            if prons:
                return self._build(random.choice(prons), lines, target, mentions, pool)
            if fallback is None:
                fallback = (random.choice(cands), lines, target, mentions, pool)
        return self._build(*fallback)

    def _build(self, q, lines, target, mentions, pool):
        sid = q['sent'] + 1
        if q['mode'] == PRONOUN:
            prev_names = sorted(m['entity'].name for m in mentions
                                if m['sent'] == q['sent'] - 1)
            g = 'female' if target.gender == 'f' else 'male'
            cot = (f"s{sid} pron '{q['surface']}' | "
                   f"s{sid-1}: {{{', '.join(prev_names)}}} | "
                   f"unique {g} → {target.name}")
        else:
            surf = q['surface'][4:] if q['surface'].startswith('the ') else q['surface']
            parts = surf.split()
            role, attrs = parts[-1], parts[:-1]
            matches = sorted(x.name for x in pool
                             if x.role == role
                             and all(a in x.attrs for a in attrs))
            filt = f"role={role}" + (f", attrs={list(attrs)}" if attrs else "")
            cot = (f"s{sid} desc '{q['surface']}' | {filt} | "
                   f"{{{', '.join(matches)}}} → {target.name}")
        meta = edict({
            'sentences':    "\n".join(lines),
            'q_sentence':   sid,
            'q_position':   q['pos'],
            'q_expression': q['surface'],
            'cot':          cot,
        })
        return Problem(metadata=meta, answer=target.name)

    def prompt(self, metadata):
        return (
            f"{metadata['sentences']}\n\n"
            f"In sentence {metadata['q_sentence']}, what does the "
            f"{metadata['q_position']} expression "
            f"'{metadata['q_expression']}' refer to?\n"
            f"The answer is the name of the person it refers to."
        )

    def score_answer(self, answer, entry):
        norm = lambda s: (str(s or '').strip().strip('.').strip("'\"").split() or [''])[-1].lower()
        return float(norm(answer) == norm(entry.answer))