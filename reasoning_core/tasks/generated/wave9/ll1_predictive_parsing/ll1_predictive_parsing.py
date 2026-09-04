import random
from dataclasses import dataclass, field

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'll1_predictive_parsing (draw 1 of 1)',
 'hypothesis': 'HV-060',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/ll1_predictive_parsing',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1742980165,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class LL1ParsingConfig(Config):
    n_prod: int = 4
    n_expand: int = 3
    max_apply: int = 10

    def apply_difficulty(self, level):
        self.n_prod = sround(self.n_prod + level)
        self.n_expand = sround(self.n_expand + level)
        self.max_apply = sround(self.max_apply + 3 * level)


def _ll1_step(table, stack, terminal):
    """Return expand/accept/error for one LL(1) step.

    table: dict (A, a) -> production string
    stack: list of symbols, top of stack at the end
    terminal: current input terminal (or None for end marker)
    Returns (kind, detail) where kind in {'match','expand','accept','error'}.
    """
    if not stack:
        if terminal is None:
            return ('accept', None)
        return ('error', ('unexpected input', terminal))
    top = stack[-1]
    if top.islower() or top == '$':
        if terminal is not None and top == terminal:
            return ('match', terminal)
        if top == '$' and terminal is None:
            return ('accept', None)
        return ('error', ('mismatch', top, terminal))
    a = terminal if terminal is not None else '$'
    key = (top, a)
    if key in table:
        return ('expand', table[key])
    return ('error', ('no table entry', key))


def _trace(table, start, input_syms):
    """Run the deterministic LL(1) parser. Return (trace, outcome)."""
    stack = ['$', start]
    trace = []
    position = 0
    steps = 0
    max_steps = 100
    while steps < max_steps:
        terminal = input_syms[position] if position < len(input_syms) else None
        kind, detail = _ll1_step(table, stack, terminal)
        if kind == 'match':
            trace.append('match ' + terminal)
            stack.pop()
            position += 1
        elif kind == 'expand':
            prod = detail
            trace.append('expand ' + stack[-1] + ' -> ' + prod)
            stack.pop()
            if prod != 'eps':
                for sym in reversed(prod.split()):
                    stack.append(sym)
        elif kind == 'accept':
            trace.append('accept')
            return trace, 'accept'
        elif kind == 'error':
            if detail[0] == 'no table entry':
                return trace, 'fail ' + detail[1][0] + ' on ' + str(detail[1][1])
            if detail[0] == 'mismatch':
                return trace, 'fail mismatch ' + str(detail[1]) + ' on ' + str(terminal)
            return trace, 'fail unexpected ' + str(detail[1])
        steps += 1
    return trace, 'reject-steps'


def _is_ll1(table, nonterminals, terminals):
    """Rough sanity: ensure each cell has at most one production."""
    all_pairs = set()
    for key in table:
        if key in all_pairs:
            return False
        all_pairs.add(key)
    return True


def _gen_instance(rng, n_prod, n_expand, max_apply):
    # Nonterminals: A, B, C, ...
    nts = ['A', 'B', 'C', 'D', 'E'][:n_prod]
    terminals = ['a', 'b', 'c', 'd', 'e'][:n_prod + 1]
    terminals = terminals[:max(2, n_prod - 1)]

    for _attempt in range(200):
        table = {}
        # Build productions as strings of terminal/nonterminal symbols.
        starts = rng.choices(nts, k=n_prod)
        for nt in nts:
            choices = []
            # each nonterminal has 1-2 productions
            nchoices = rng.randint(1, 2)
            for _c in range(nchoices):
                length = rng.randint(0, 2)
                syms = []
                for _s in range(length):
                    if rng.random() < 0.5:
                        syms.append(rng.choice(terminals))
                    else:
                        cand = rng.choice(nts)
                        # avoid trivial self-loops when possible
                        syms.append(cand)
                prod = 'eps' if not syms else ' '.join(syms)
                choices.append(prod)
            # assign productions; use first-symbol to map to terminals for LL1
            seen_first = set()
            for prod in choices:
                first = prod.split()[0] if prod != 'eps' else 'eps'
                # eps productions can be assigned to the '$' lookahead
                term = rng.choice(terminals + ['$'])
                if term in seen_first:
                    continue
                seen_first.add(term)
                table[(nt, term)] = prod
        if not _is_ll1(table, nts, terminals):
            continue

        start = rng.choice(nts)
        # Build a valid input by expanding from start using productions
        # (a leftmost derivation grown forward, targeting n_expand expansions).
        input_syms = _expand_from(rng, table, start, nts, terminals, max_apply, n_expand)
        if input_syms is None:
            continue
        if len(input_syms) > 20:
            continue
        trace, outcome = _trace(table, start, input_syms)
        if outcome != 'accept' or len(trace) > 40:
            continue
        # ensure the trace is non-trivial and answer variable
        return table, start, input_syms, trace, nts, terminals

    raise RuntimeError('could not build LL1 instance')


def _expand_from(rng, table, start, nts, terminals, max_apply, n_expand):
    """Build a concrete input string that the LL1 parser accepts.

    Grows a genuine leftmost derivation (stack top-of-stack = next symbol),
    preferring nonterminal-bearing productions so the resulting traces tend to
    have real depth rather than collapsing to a single terminal.
    """
    work = [start]
    out = []
    applies = 0
    for _step in range(2000):
        if not work:
            break
        sym = work.pop(0)
        if sym in nts:
            applies += 1
            if applies > max_apply:
                return None
            keys = [k for k in table if k[0] == sym]
            if not keys:
                return None
            # Prefer productions whose body actually contains a nonterminal so
            # the derivation can reach real depth.
            nont_choices = [k for k in keys if table[k] != 'eps'
                            and any(s in nts for s in table[k].split())]
            if applies < n_expand and nont_choices:
                pool = nont_choices
            else:
                pool = keys
            a = rng.choice(pool)[1]
            prod = table[(sym, a)]
            if prod == 'eps':
                continue
            syms = prod.split()
            work = syms + work
        else:
            out.append(sym)
    if applies > max_apply or not out:
        return None
    return out


class LL1PredictiveParsing(Task):
    summary = "Execute deterministic LL(1) stack parsing from a supplied parse table, returning the production trace or exact failure point."
    config_cls = LL1ParsingConfig
    task_version = 2

    def generate_entry(self):
        c = self.config
        rng = random
        table, start, input_syms, trace, nts, terminals = _gen_instance(rng, c.n_prod, c.n_expand, c.max_apply)

        # Build a readable parse table payload.
        rows = []
        for (nt, term) in sorted(table.keys()):
            rows.append([nt, term, table[(nt, term)]])
        input_str = ' '.join(input_syms)

        # reference trace of expand operations only + outcome
        trace_str = ' ; '.join(trace)

        metadata = edict({
            'table': rows,
            'start': start,
            'input': input_str,
            'terminals': sorted(set(terminals)),
            'nonterminals': nts,
            'trace': trace_str,
        })
        metadata.payload = {
            'nonterminals': nts,
            'terminals': sorted(set(terminals)),
            'start': start,
            'table': rows,
            'input': input_str,
        }

        # Answer: the trace (production trace) or failure point.
        answer = trace_str

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        lines = []
        lines.append('Nonterminals: ' + ', '.join(payload['nonterminals']))
        lines.append('Terminals: ' + ', '.join(payload['terminals']))
        lines.append('Start symbol: ' + payload['start'])
        lines.append('Parse table (rows "A b -> production", "eps" means empty production):')
        for row in payload['table']:
            lines.append('  %s %s -> %s' % (row[0], row[1], row[2]))
        lines.append('Input string to parse: ' + payload['input'])
        lines.append('')
        lines.append('Run the deterministic LL(1) predictive parser using a stack that starts as '
                     '[$, start symbol] with $ the end-of-input marker and the top of stack on the '
                     'right. The next input token is read left to right. Standard LL(1) algorithm: '
                     'if the top of stack is a terminal (or $), match it against the current input '
                     'token and pop; if it is a nonterminal, look up the parse table cell '
                     '(top-nonterminal, current-input-token) where $ stands in for end of input, '
                     'and expand: pop the nonterminal and push the production symbols in reverse '
                     'order (an "eps" production pushes nothing).')
        lines.append('Report every parse-table expansion as "expand X -> prod" and every successful '
                     'terminal match as "match t", in execution order, ending with "accept". If at '
                     'any point the current token does not match the top of stack, or the parse '
                     'table has no entry (top-nonterminal, current-token), stop immediately and '
                     'report the exact failure point as "fail no-entry X on t" or '
                     '"fail mismatch X on t" (no "accept" is appended in the failure case).')
        lines.append('')
        lines.append('The answer is that exact trace string.')
        return '\n'.join(lines)

    def score_answer(self, answer, entry):
        ground = entry.answer
        if not isinstance(answer, str):
            return 0.0
        # Accept exact or optionally exact with whitespace normalization.
        normalized = ' '.join(answer.split())
        ground_norm = ' '.join(ground.split())
        if normalized == ground_norm:
            return 1.0
        return 0.0
