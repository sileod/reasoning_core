import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'petri_conflict_pair (draw 2 of 2)',
 'hypothesis': 'W1-078',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/petri_conflict_pair',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1253355878,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def enabled(trans, place_tokens):
    for (p, w) in trans['inputs']:
        if place_tokens[p] < w:
            return False
    return True


def conflicting_pairs(transitions, place_tokens):
    """Return list of (i, j, p_idx, deficit) for every enabled conflicting pair
    i<j that share input place p_idx. deficit = wa+wb-tokens[p] > 0."""
    res = []
    n = len(transitions)
    for i in range(n):
        for j in range(i + 1, n):
            ti = transitions[i]
            tj = transitions[j]
            if not (enabled(ti, place_tokens) and enabled(tj, place_tokens)):
                continue
            inp_i = dict(ti['inputs'])
            inp_j = dict(tj['inputs'])
            for p in inp_i:
                if p in inp_j:
                    deficit = inp_i[p] + inp_j[p] - place_tokens[p]
                    if deficit > 0:
                        res.append((i, j, p, deficit))
    return res


def parse_answer(answer):
    text = str(answer).replace(' ', '').strip()
    if text == '':
        return None
    if text.lower() == 'none':
        return ('NONE',)
    parts = text.split(',')
    if len(parts) != 4:
        return None
    ti, tj, pname, defstr = parts
    if not (ti.startswith('t') and tj.startswith('t') and pname.startswith('p')):
        return None
    if not (defstr.lstrip('-').isdigit()):
        return None
    return (ti, tj, pname, defstr)


class PetriConflictPairConfig(Config):
    n_places: int = 4
    n_trans: int = 3
    max_tok: int = 5
    max_wt: int = 4
    none_prob: float = 0.2

    def apply_difficulty(self, level):
        self.n_places = sround(self.n_places + level)
        self.n_trans = sround(self.n_trans + level)
        self.max_tok = sround(self.max_tok + level)
        self.max_wt = sround(self.max_wt + min(level, 2))
        if self.n_places < self.n_trans:
            self.n_places = self.n_trans


class PetriConflictPair(Task):
    summary = "Output the lexicographically smallest pair of individually enabled Petri-net transitions that share an input place whose tokens are insufficient for both (carrying the shared place and token deficit), or None when no such conflict exists."
    config_cls = PetriConflictPairConfig

    def generate_entry(self):
        c = self.config
        n_places = int(c.n_places)
        n_trans = int(c.n_trans)
        max_tok = int(c.max_tok)
        max_wt = int(c.max_wt)

        if n_trans < 2:
            n_trans = 2
        if n_places < n_trans:
            n_places = n_trans

        place_names = ['p%d' % (i + 1) for i in range(n_places)]
        trans_names = ['t%d' % (i + 1) for i in range(n_trans)]

        none_mode = random.random() < float(c.none_prob)
        transitions = None
        answer = None

        for _ in range(200):
            if none_mode:
                place_tokens = {p: random.randint(1, max_tok) for p in place_names}
                transitions = []
                for i in range(n_trans):
                    p = place_names[i]
                    w = random.randint(1, min(place_tokens[p], max_wt))
                    transitions.append({'name': trans_names[i], 'inputs': [(p, w)]})
                conflicts = conflicting_pairs(transitions, place_tokens)
                if not conflicts:
                    answer = 'None'
                    break
            else:
                place_tokens = {p: random.randint(1, max_tok) for p in place_names}
                hub = random.randrange(n_places)
                k = min(place_tokens[place_names[hub]], max_wt)
                place_tokens[place_names[hub]] = k
                wa = random.randint(1, min(k, max_wt))
                wb = random.randint(max(1, k - wa + 1), min(k, max_wt))
                transitions = []
                for i in range(n_trans):
                    trans = {'name': trans_names[i], 'inputs': []}
                    if i == 0:
                        trans['inputs'].append((place_names[hub], wa))
                        for _ in range(random.randint(0, 1)):
                            op = place_names[random.randrange(n_places)]
                            if op != place_names[hub] and place_tokens[op] >= 1:
                                trans['inputs'].append((op, 1))
                    elif i == 1:
                        trans['inputs'].append((place_names[hub], wb))
                        for _ in range(random.randint(0, 1)):
                            op = place_names[random.randrange(n_places)]
                            if op != place_names[hub] and place_tokens[op] >= 1:
                                trans['inputs'].append((op, 1))
                    else:
                        n_inp = random.randint(2, min(3, n_places))
                        chosen = random.sample(place_names, n_inp)
                        for p in chosen:
                            trans['inputs'].append((p, random.randint(1, min(place_tokens[p], max_wt))))
                    transitions.append(trans)
                conflicts = conflicting_pairs(transitions, place_tokens)
                if not conflicts:
                    continue  # conflict mode must contain at least the constructed pair
                # canonical: smallest (i, j); for that pair the shared place
                # with the largest deficit (tie-break smallest place index)
                best = min(conflicts, key=lambda c_: (c_[0], c_[1]))
                pair_conf = [x for x in conflicts if x[0] == best[0] and x[1] == best[1]]
                ph = min(pair_conf, key=lambda x: (-x[3], x[2]))
                i, j, p_idx, deficit = ph
                # verify individually enabled and deficit >= 1
                assert enabled(transitions[i], place_tokens)
                assert enabled(transitions[j], place_tokens)
                assert deficit >= 1
                answer = '%s,%s,%s,%d' % (trans_names[i], trans_names[j],
                                          p_idx, deficit)
                break

        if answer is None:
            raise RuntimeError('failed to construct petri net instance')

        metadata = edict({
            'places': [{'name': p, 'tokens': int(place_tokens[p])} for p in place_names],
            'transitions': [{'name': t['name'],
                             'inputs': [[p, int(w)] for (p, w) in t['inputs']]}
                            for t in transitions],
            'conflicts': [[trans_names[i], trans_names[j], p,
                           int(d)] for (i, j, p, d) in sorted(conflicts,
                           key=lambda x: (x[0], x[1]))],
        })
        metadata.payload = {
            'places': [{'name': p, 'tokens': int(place_tokens[p])} for p in place_names],
            'transitions': [{'name': t['name'],
                             'inputs': [[p, int(w)] for (p, w) in t['inputs']]}
                            for t in transitions],
        }
        if answer == 'None':
            assert not conflicts
        else:
            assert conflicts
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        lines.append('A small Petri net has the following places with their token counts:')
        for pl in metadata.places:
            lines.append('- %s: %d tokens' % (pl['name'], pl['tokens']))
        lines.append('')
        lines.append('Each transition lists its input places as pairs (place, weight):')
        for tr in metadata.transitions:
            pairs = ', '.join('%s/%d' % (p, w) for (p, w) in tr['inputs'])
            lines.append('- %s: consumes %s' % (tr['name'], pairs))
        lines.append('')
        lines.append('A transition is enabled if every input place holds at least its weight. '
                     'Two enabled transitions with distinct names are a conflicting pair if they '
                     'share some input place whose token count is less than the sum of the two '
                     'weights on it (so firing both is impossible). The deficit of that shared '
                     'place is the sum of the two weights minus its token count.')
        lines.append('')
        lines.append('Among all conflicting pairs, take the one whose two transition names are '
                     'lexicographically smallest (the pair with the smallest first name, then the '
                     'smallest second name; names sort as t1 < t10 < t2). If that pair conflicts '
                     'over several shared places, use the place with the largest deficit '
                     '(ties to the place with the smallest name).')
        body = '\n'.join(lines)
        return (body + '\n\nAnswer '
                'as None if there is no conflicting pair, otherwise as '
                'firstTransition,secondTransition,sharedPlace,deficit '
                '(for example "t2,t5,p3,2" would mean t2 and t5 conflict '
                'over p3 which is short by 2 tokens).')

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        parsed = parse_answer(answer)
        if parsed is None:
            return 0.0
        expected = parse_answer(gold)
        if expected is None:
            return 0.0
        if parsed == expected:
            return 1.0
        return 0.0
