import random
from dataclasses import dataclass, field

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'temporal_logic_monitoring (draw 1 of 1)',
 'hypothesis': 'HV-056',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/temporal_logic_monitoring',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 43127990,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def eval_formula(formula, atom_true, pos):
    """Evaluate a finite-trace temporal formula at position pos of a trace.

    Trace positions run 0..T-1 (T = len of every atom_true list).
    Semantics (standard finite-trace LTL, evaluated within the trace):
      X  phi      : phi holds at pos+1 (false at the last position)
      F  phi      : phi holds at some position >= pos
      G  phi      : phi holds at every position >= pos
      phi U psi   : psi holds at some j >= pos and phi holds at all k in [pos, j)
    """
    T = len(atom_true[list(atom_true.keys())[0]])
    op = formula[0]
    if op == 'atom':
        return atom_true[formula[1]][pos]
    if op == 'not':
        return not eval_formula(formula[1], atom_true, pos)
    if op == 'and':
        return eval_formula(formula[1], atom_true, pos) and eval_formula(formula[2], atom_true, pos)
    if op == 'or':
        return eval_formula(formula[1], atom_true, pos) or eval_formula(formula[2], atom_true, pos)
    if op == 'X':
        if pos + 1 >= T:
            return False
        return eval_formula(formula[1], atom_true, pos + 1)
    if op == 'F':
        return any(eval_formula(formula[1], atom_true, j) for j in range(pos, T))
    if op == 'G':
        return all(eval_formula(formula[1], atom_true, j) for j in range(pos, T))
    if op == 'U':
        left, right = formula[1], formula[2]
        for j in range(pos, T):
            if eval_formula(right, atom_true, j):
                if all(eval_formula(left, atom_true, k) for k in range(pos, j)):
                    return True
        return False
    raise ValueError('unknown operator: %r' % (op,))


def evaluate_trace(formula, atom_true):
    """Top-level truth value (boolean) of formula at each trace position."""
    T = len(atom_true[list(atom_true.keys())[0]])
    return [eval_formula(formula, atom_true, p) for p in range(T)]


def build_formula(depth, atoms, rng):
    """Build a formula tuple over the given atoms with target questcap depth.

    Uses next (X), eventually (F), always (G), until (U) and Boolean and/or/not
    connectors. depth ~= number of operator levels.
    """
    if depth <= 0:
        return ('atom', rng.choice(atoms))

    kind = rng.random()
    if kind < 0.22:
        return ('not', build_formula(depth - 1, atoms, rng))
    if kind < 0.44:
        return ('X', build_formula(depth - 1, atoms, rng))
    if kind < 0.62:
        return ('F', build_formula(depth - 1, atoms, rng))
    if kind < 0.78:
        return ('G', build_formula(depth - 1, atoms, rng))
    if kind < 0.90:
        return ('and', build_formula(depth - 1, atoms, rng), build_formula(depth - 1, atoms, rng))
    if kind < 0.96:
        return ('or', build_formula(depth - 1, atoms, rng), build_formula(depth - 1, atoms, rng))
    return ('U', build_formula(depth - 1, atoms, rng), build_formula(depth - 1, atoms, rng))


ATOM_NAMES = ['a', 'b', 'c']


def render_formula(formula):
    op = formula[0]
    if op == 'atom':
        return formula[1]
    if op == 'not':
        return '!\u00a0(%s)' % render_formula(formula[1])
    if op == 'and':
        return '(%s) /\\(%s)' % (render_formula(formula[1]), render_formula(formula[2]))
    if op == 'or':
        return '(%s) \\/(%s)' % (render_formula(formula[1]), render_formula(formula[2]))
    if op == 'X':
        return 'X\u00a0(%s)' % render_formula(formula[1])
    if op == 'F':
        return 'F\u00a0(%s)' % render_formula(formula[1])
    if op == 'G':
        return 'G\u00a0(%s)' % render_formula(formula[1])
    if op == 'U':
        return '(%s) U (%s)' % (render_formula(formula[1]), render_formula(formula[2]))
    raise ValueError(formula)


@dataclass
class TemporalLogicConfig(Config):
    length: int = 6
    atoms: int = 2
    depth: int = 2

    def apply_difficulty(self, level):
        self.length = int(sround(6 + level * 2))
        self.atoms = 2 if level < 3 else 3
        self.depth = int(sround(2 + level))


class TemporalLogicMonitoring(Task):
    summary = ("Evaluate finite-trace LTL formulas built from next, eventually, always, "
               "and until over proposition traces, returning the sorted list of every trace "
               "position where the top-level formula is true.")
    config_cls = TemporalLogicConfig

    def generate_entry(self):
        cfg = self.config
        rng = random
        length = cfg.length
        n_atoms = cfg.atoms
        depth = cfg.depth

        atoms = ATOM_NAMES[:n_atoms]

        for _attempt in range(200):
            atom_true = {}
            for name in atoms:
                atom_true[name] = [rng.random() < 0.5 for _ in range(length)]

            formula = build_formula(depth, atoms, rng)
            truths = evaluate_trace(formula, atom_true)
            true_positions = sorted(p for p, t in enumerate(truths) if t)

            if len(true_positions) == 0 or len(true_positions) == length:
                continue

            # Independent structural verification: re-derive every claimed and
            # every rejected position from the formula and trace.
            ok = True
            for p, t in enumerate(truths):
                if eval_formula(formula, atom_true, p) != t:
                    ok = False
                    break
            if not ok:
                continue

            answer = ','.join(str(x) for x in true_positions)
            metadata = edict({
                'length': length,
                'atoms': atoms,
                'trace': {name: ''.join('1' if b else '0' for b in atom_true[name])
                          for name in atoms},
                'formula': render_formula(formula),
                'answer_positions': true_positions,
            })
            metadata.payload = {
                'trace': metadata.trace,
                'formula': metadata.formula,
            }
            return Entry(metadata=metadata, answer=answer)

        raise RuntimeError('could not produce a valid non-degenerate instance')

    def render_prompt(self, metadata):
        lines = ['Time positions are the integers 0, 1, ..., %d.' % (metadata.length - 1),
                 'Each row shows, per position, whether that proposition holds (1) or not (0):']
        for name in metadata.payload['trace']:
            lines.append('  %s: %s' % (name, metadata.payload['trace'][name]))
        lines.append('Evaluate the temporal formula')
        lines.append('  %s' % metadata.payload['formula'])
        lines.append('over the trace above. Next (X) moves one step forward within the trace; '
                     'eventually (F) requires holding at some current-or-later position; '
                     'always (G) requires holding at every current-or-later position; '
                     'until (U) requires the right formula to hold at some later-or-equal position '
                     'with the left formula holding at every position strictly before it.')
        lines.append('The answer is the sorted, comma-separated list of all positions where the '
                     'formula is true; if it is true nowhere, the answer is the single word "none".')
        return '\n'.join(lines)

    def score_answer(self, answer, entry):
        gold = sorted(entry.metadata.answer_positions)
        if answer is None:
            return 0.0
        text = str(answer).strip()
        if text.lower() in ('none', '[],', '[]', '()', 'nothing'):
            if len(gold) == 0:
                return 1.0
            return 0.0
        nums = []
        text = text.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                nums.append(int(part))
            except ValueError:
                return 0.0
        nums = sorted(set(nums))
        if nums == gold:
            return 1.0
        return 0.0
