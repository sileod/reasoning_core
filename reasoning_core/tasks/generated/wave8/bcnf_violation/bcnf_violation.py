"""BCNF violation detection.

Given a schema (attribute set) and a set of functional dependencies, find the
first canonical BCNF violation and output it, or None if the schema is in BCNF.

Canonical form: violations are reported as (X, A) where X is the dependency's
determinant in canonical (sorted, deduplicated) order and A is the attribute
it determines, with A not already in X. The "first" violation is determined by
the dependency scan order defined in generation and is stable.
"""

from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'bcnf_violation (draw 2 of 2)',
 'hypothesis': 'W1-029',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/bcnf_violation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4052403849,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

import random


@dataclass
class BcnfConfig(Config):
    n_attrs: int = 6
    n_deps: int = 6
    attr_range: int = 6
    ensure_violation: float = 0.8

    def apply_difficulty(self, level):
        self.n_attrs = sround(self.n_attrs + level)
        self.n_deps = sround(self.n_deps + level)
        self.attr_range = self.n_attrs


def _closure(attrs, deps, universe):
    """Return the attribute closure of frozenset attrs over deps within universe."""
    current = set(attrs)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in deps:
            if lhs.issubset(current):
                new = rhs - current
                if new:
                    current |= new
                    changed = True
    return current & universe


def _is_superkey(attrs, deps, universe):
    return _closure(attrs, deps, universe) == universe


def _first_violation(attrs_list, deps, universe, order):
    """Return (X, A) canonical first violation or None.

    order is the list of deps dictating scan order. For each dep (lhs, rhs) in order,
    check each A in rhs not in lhs (sorted). A violation is when lhs is not a
    superkey. Return first (canonical lhs, A).
    """
    for lhs, rhs in order:
        if _is_superkey(lhs, deps, universe):
            continue
        for a in sorted(rhs - lhs):
            return (tuple(sorted(lhs)), a)
    return None


class BcnfViolation(Task):
    summary = ("Given a relational schema and functional dependencies, output the "
               "first canonical BCNF violation (determinant, attribute) or None; "
               "violations and BCNF-valid schemas both occur, determinants vary in "
               "size and attributes vary in name and range.")
    config_cls = BcnfConfig
    task_version = 2

    def _attr_names(self, n):
        return [chr(ord('A') + i) for i in range(n)]

    def generate_entry(self):
        cfg = self.config
        universe = set(range(cfg.attr_range))
        names = self._attr_names(cfg.attr_range)
        ensure_violation = random.random() < cfg.ensure_violation

        n_attrs = cfg.attr_range
        n_deps = cfg.n_deps

        if ensure_violation:
            # Build mostly-random deps and force at least one non-superkey determinant.
            deps_set = set()
            attempts = 0
            while len(deps_set) < n_deps and attempts < n_deps * 40:
                attempts += 1
                n_lhs = random.randint(1, max(1, n_attrs - 1))
                lhs = frozenset(random.sample(range(n_attrs), n_lhs))
                candidates = [a for a in range(n_attrs) if a not in lhs]
                if not candidates:
                    continue
                a = random.choice(candidates)
                dep = (lhs, frozenset({a}))
                if dep not in deps_set:
                    deps_set.add(dep)
            deps = list(deps_set)
            random.shuffle(deps)
            violation = _first_violation([d[0] for d in deps], deps, universe, deps)
            if violation is None:
                # Force a violation: add a non-superkey determinant dep.
                non_super = [d for d in deps if not _is_superkey(d[0], deps, universe)]
                if non_super:
                    lhs = non_super[0][0]
                else:
                    opts = [frozenset({x}) for x in range(n_attrs)
                            if not _is_superkey(frozenset({x}), deps, universe)]
                    lhs = random.choice(opts)
                cands = [a for a in range(n_attrs) if a not in lhs]
                na = random.choice(cands)
                deps.append((lhs, frozenset({na})))
                random.shuffle(deps)
                violation = _first_violation([d[0] for d in deps], deps, universe, deps)
        else:
            # Build a guaranteed BCNF schema: choose a key K and all deps have lhs a
            # superset of K (hence a superkey since K determines everything).
            k_size = random.randint(1, max(1, n_attrs - 1))
            key = frozenset(random.sample(range(n_attrs), k_size))
            # Key -> each attribute outside the key guarantees the key is a superkey.
            deps_set = set()
            for a in range(n_attrs):
                if a not in key:
                    deps_set.add((key, frozenset({a})))
                    if len(deps_set) >= n_deps:
                        break
            # If we still need more deps, add deps whose lhs is a superset of key
            # (still superkeys).
            attempts = 0
            while len(deps_set) < n_deps and attempts < n_deps * 30:
                attempts += 1
                lhs = frozenset(key)
                base = set(key)
                add = base
                n_lhs = random.randint(k_size + 1, min(n_attrs, k_size + 2))
                extras = random.sample([x for x in range(n_attrs) if x not in base],
                                       n_lhs - len(base))
                add = base.union(extras)
                lhs = frozenset(add)
                cands = [a for a in range(n_attrs) if a not in lhs]
                if not cands:
                    continue
                na = random.choice(cands)
                deps_set.add((lhs, frozenset({na})))
                # key with a fresh rhs too
                keycands = [a for a in range(n_attrs) if a not in key]
                if keycands:
                    deps_set.add((frozenset(key), frozenset({random.choice(keycands)})))
            deps = list(deps_set)
            random.shuffle(deps)
            violation = _first_violation([d[0] for d in deps], deps, universe, deps)
            if violation is not None:
                # Should not happen for a clean BCNF build; fall back by rejecting.
                raise RuntimeError("BNCF build unexpectedly produced a violation")

        # domain sanity: violation is a (lhs, attr) with attr not in lhs, lhs non-superkey
        if violation is not None:
            lhs, a = violation
            assert a not in lhs
            assert not _is_superkey(frozenset(lhs), deps, universe)

        # Build prompt text
        dep_lines = []
        for (lhs, rhs) in deps:
            lhs_s = "{" + ", ".join(names[x] for x in sorted(lhs)) + "}"
            rhs_s = "{" + ", ".join(names[x] for x in sorted(rhs)) + "}"
            dep_lines.append(f"{lhs_s} -> {rhs_s}")

        schema = f"R({', '.join(names)})"

        if violation is None:
            answer = "None"
        else:
            lhs, a = violation
            answer = "{" + ", ".join(names[x] for x in lhs) + "} -> " + names[a]

        metadata = edict({
            "schema": schema,
            "dependencies": dep_lines,
            "n_attrs": cfg.attr_range,
            "answer": answer,
        })
        metadata.payload = {
            "schema": schema,
            "dependencies": dep_lines,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        deps_str = "; ".join(metadata.payload["dependencies"])
        return (
            f"Consider the relational schema {metadata.payload['schema']} with functional "
            f"dependencies: {deps_str}. Determine whether the schema is in Boyce-Codd "
            f"Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial "
            f"functional dependency has a superkey as its determinant. If the schema is "
            f"NOT in BCNF, state the FIRST BCNF violation as the functional dependency "
            f"X -> A, where X is the non-superkey determinant, A is the determined "
            f"attribute not in X, attributes within a set are listed in alphabetical "
            f"order, and each set is written as {{A, C}}. If the schema IS in BCNF, the "
            f"answer is the word None.\n\n"
            f"What is the first BCNF violation of this schema?"
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 1.0 if gold == "None" else 0.0
        if not isinstance(answer, str):
            return 0.0
        s = answer.strip()
        return 1.0 if s == gold else 0.0
