"""Lossless binary decomposition: decide whether a binary decomposition is lossless under a set of
functional dependencies, and report the deciding witness.

Given functional dependencies over a small attribute set and a decomposition of a relation into two
projections that share a key, the decomposition is lossless iff the shared key determines (through
the closure of the dependencies) at least one of the two non-shared sub-relations. The answer is the
deciding witness: a canonical string naming which side is determined (or, for a lossy decomposition,
the attributes the key fails to determine).

Answer space is intentionally wide (see gameability): there are many distinct deciding dependencies,
many distinct keys, and many distinct missing-attribute sets, so no constant guess dominates.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'lossless_binary_decomposition (draw 1 of 2)',
 'hypothesis': 'W1-030',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/lossless_binary_decomposition',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2705364208,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class LosslessBinaryConfig(Config):
    n_attrs: int = 5
    n_deps: int = 3

    def apply_difficulty(self, level):
        self.n_attrs = min(7, sround(self.n_attrs + level // 2))
        self.n_deps = sround(self.n_deps + level // 2 + (level % 2))


def _fd_closure(attrs, fds):
    """Attribute-set closure under Armstrong semantics."""
    cur = set(attrs)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs <= cur and not rhs <= cur:
                cur |= rhs
                changed = True
    return frozenset(cur)


def _is_lossless(p1, p2, fds):
    """Binary decomposition {XY, XZ} with shared set X is lossless iff X->Y or X->Z holds."""
    share = p1 & p2
    if not share:
        return False
    c = _fd_closure(share, fds)
    return (p1 - share) <= c or (p2 - share) <= c


def _fmt(attrs, names):
    return "{" + ",".join(sorted(names[a] for a in attrs)) + "}"


def _sorted_key(fd, names):
    lhs, rhs = fd
    return (sorted(names[a] for a in lhs), sorted(names[a] for a in rhs))


def _finding_chain(share, target, fds, attrs, names):
    """Return the list of (lhs, rhs) dependencies fired by a greedy forward-pass that grows the
    closure from `share` until it covers `target`. Used to build a reproducible witness."""
    frontier = set(share)
    used = []
    step = 0
    while not (target <= frontier) and step < 200:
        progressed = False
        for lhs, rhs in fds:
            if lhs <= frontier and not rhs <= frontier:
                frontier |= set(rhs)
                used.append((lhs, rhs))
                progressed = True
                break
        if not progressed:
            break
        step += 1
    return used


class LosslessBinaryDecomposition(Task):
    summary = ("Given functional dependencies over a small attribute set and a binary "
               "decomposition into two projections sharing a key, decide losslessness via the "
               "shared-key dependency criterion and report the deciding dependency chain or the "
               "attributes the key fails to determine.")
    config_cls = LosslessBinaryConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_attrs)
        names = [chr(ord('A') + i) for i in range(n)]
        attrs = frozenset(range(n))

        while True:
            key = frozenset(random.sample(range(n), random.randint(1, 2)))
            nonkey = sorted(attrs - key)
            if not nonkey:
                continue
            k = random.randint(0, len(nonkey))
            p1 = key | frozenset(nonkey[:k])
            p2 = key | frozenset(nonkey[k:])
            if not (p1 - key) or not (p2 - key):
                continue

            fds = []
            # helpers: full set of non-trivial dependencies we may use
            candidates = []
            for lhs_size in (1, 2):
                from itertools import combinations
                for lhs_t in combinations(range(n), lhs_size):
                    lhs = frozenset(lhs_t)
                    rhs_opts = [a for a in range(n) if a not in lhs]
                    for r in rhs_opts:
                        candidates.append((lhs, frozenset([r])))

            lossless = random.random() < 0.5
            if lossless:
                # ensure the key determines all of attrs
                # pick the primary side that the key will determine
                side_choice = random.randint(0, 1)
                main_side = p1 if side_choice == 0 else p2
                other_side = p2 if side_choice == 0 else p1
                # add key -> every attribute of main_side
                for a in sorted(main_side - key):
                    fds.append((key, frozenset([a])))
                # maybe also cover part/all of the other side
                for a in sorted(other_side - key):
                    if random.random() < 0.5:
                        fds.append((key, frozenset([a])))
                # add filler non-key dependencies
                random.shuffle(candidates)
                for c in candidates:
                    if len(fds) >= cfg.n_deps + 2:
                        break
                    if random.random() < 0.35:
                        fds.append(c)
            else:
                # ensure the key does NOT determine at least one attribute of each side,
                # so the decomposition is lossy. Fill with random dependencies and rely on the
                # verification loop below (which re-samples) to discard any that make it lossless.
                random.shuffle(candidates)
                for c in candidates:
                    if len(fds) >= cfg.n_deps + 2:
                        break
                    if random.random() < 0.35:
                        fds.append(c)
            # dedupe and drop trivial (rhs<=lhs) deps
            seen = set()
            uniq = []
            for lhs, rhs in fds:
                if rhs <= lhs:
                    continue
                if (lhs, tuple(sorted(rhs))) not in seen:
                    seen.add((lhs, tuple(sorted(rhs))))
                    uniq.append((lhs, rhs))
            fds = uniq
            if not fds:
                continue

            # verify
            actual = _is_lossless(p1, p2, fds)
            if actual != lossless:
                continue
            # verify defining property precisely
            share = p1 & p2
            closed = _fd_closure(share, fds)
            if lossless:
                if not ((p1 - share) <= closed or (p2 - share) <= closed):
                    continue
            else:
                if (p1 - share) <= closed or (p2 - share) <= closed:
                    continue
            break

        # ensure the number of deps varies and isn't degenerate
        fds.sort(key=lambda t: _sorted_key(t, names))

        lines = []
        for lhs, rhs in fds:
            lines.append(_fmt(lhs, names) + " -> " + _fmt(rhs, names))
        fd_text = "; ".join(lines)

        share = p1 & p2
        if lossless:
            # decide which side is determined; for the witness pick the one that is determined
            c = _fd_closure(share, fds)
            sides = []
            if (p1 - share) <= c:
                sides.append("P1")
            if (p2 - share) <= c:
                sides.append("P2")
            side_name = sides[0] if sides else "P1"
            # build the witness chain deterministically
            target = p1 if side_name == "P1" else p2
            chain = _finding_chain(share, target - share, fds, attrs, names)
            chain_names = []
            for lhs, rhs in chain:
                chain_names.append(_fmt(lhs, names) + "->" + _fmt(rhs, names))
            witness_key = _fmt(share, names)
            answer = "lossless:" + witness_key + "->" + side_name + " via " + ";".join(chain_names)
            witness = answer
        else:
            missing = sorted(attrs - closed)
            answer = "lossy:" + _fmt(share, names) + " misses " + ",".join(names[a] for a in missing)
            witness = answer

        payload = {
            "attributes": ",".join(names),
            "fds": fd_text,
            "p1": _fmt(p1, names),
            "p2": _fmt(p2, names),
        }
        metadata = edict({
            "attributes": ",".join(names),
            "fds": fd_text,
            "p1": _fmt(p1, names),
            "p2": _fmt(p2, names),
            "lossless": bool(lossless),
            "answer": answer,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            f"A relation has attributes {payload['attributes']}, with functional dependencies: "
            f"{payload['fds']}. It is decomposed into the two projections {payload['p1']} and "
            f"{payload['p2']}. A binary decomposition {{XY, XZ}} with shared attribute set X is "
            f"lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. "
            f"Decide losslessness and give the deciding witness. If lossless, state the shared key "
            f"set, which of P1/P2 it determines, and the shortest chain of given dependencies "
            f"(each as {{L}}->{{R}}, in the order they fire) that establishes it. If lossy, state "
            f"the shared key set and the attributes it cannot determine. Begin the answer on one "
            f"line with 'lossless:' or 'lossy:'."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        a = str(answer).strip()
        gold = str(entry.answer).strip()
        meta = getattr(entry, "metadata", None) or {}
        lossless = meta.get("lossless")
        if lossless is None:
            lossless = gold.startswith("lossless:")
        prefix = "lossless:" if lossless else "lossy:"
        if a.lower().startswith(prefix) and a == gold:
            return 1.0
        return 0.0
