import re, random, itertools as it
from dataclasses import dataclass
from collections import defaultdict
from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


LINKS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "sigma"
]


def _atom(p, a, b):
    return (p, a, b)


def _nodes(atoms):
    return sorted({x for _, a, b in atoms for x in (a, b)})


def _preds(atoms):
    return sorted({p for p, _, _ in atoms})


def _obj_names(n, start=0):
    base = list("abcdefghijklmnopqrstuvwxyz")
    return base[start:start + n] if start + n <= len(base) else [f"o{i}" for i in range(n)]


def _query_names(n):
    base = list("xyzuvw")
    return base[:n] if n <= len(base) else [f"x{i}" for i in range(n)]


def _link_names(n, offset=0):
    return LINKS[offset:offset + n] if offset + n <= len(LINKS) else [f"rel{i}" for i in range(n)]


def _sent(atom):
    p, a, b = atom
    return f"{a} is {p}-linked to {b}."


def _compact(atom):
    p, a, b = atom
    return f"{a} {p} {b}"


def _parse_sent(s):
    s = str(s).strip()
    m = re.search(r"\b(\w+)\s+is\s+([A-Za-z]\w*)-linked\s+to\s+(\w+)\b", s)
    if m:
        a, p, b = m.groups()
        return p, a, b
    m = re.search(r"\b([A-Za-z]\w*)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", s)
    if m:
        return m.groups()
    m = re.fullmatch(r"(\w+)\s+([A-Za-z]\w*)\s+(\w+)\.?", s)
    if m:
        a, p, b = m.groups()
        return p, a, b
    return None


def _rand_atoms(nodes, preds, m, rng, avoid=()):
    avoid = set(avoid)
    pool = [(p, a, b) for p in preds for a in nodes for b in nodes if a != b and (p, a, b) not in avoid]
    rng.shuffle(pool)
    return set(pool[:m])


def _sample_param(value, value_range, rng):
    return rng.randint(*value_range) if value_range is not None else int(value)


def _weakly_connected(atoms):
    ns = _nodes(atoms)
    if not ns:
        return False
    adj = defaultdict(set)
    for _, a, b in atoms:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {ns[0]}, [ns[0]]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(ns)


def _candidate_consequences(before, nodes, preds, rng):
    pool = [(p, a, b) for p in preds for a in nodes for b in nodes if a != b and (p, a, b) not in before]
    rng.shuffle(pool)
    return pool


def _near_context(q_before, q_consequence, m, rng):
    p, u, v = q_consequence
    required = []
    for group in (
        [x for x in q_before if x[0] == p],
        [x for x in q_before if u in x[1:]],
        [x for x in q_before if v in x[1:]],
    ):
        rng.shuffle(group)
        for atom in group:
            if atom not in required:
                required.append(atom)
                break
    near = [x for x in q_before if (x[0] == p or u in x[1:] or v in x[1:]) and x not in required]
    far = [x for x in q_before if x not in near and x not in required]
    rng.shuffle(near)
    rng.shuffle(far)
    return set((required + near + far)[:m])


def _inverse_case(q_before, q_consequence, m, rng, reverse_rate=0.15):
    ctx = _near_context(q_before, q_consequence, m, rng)
    all_atoms = sorted(ctx | {q_consequence})
    qnodes, qpreds = _nodes(all_atoms), _preds(all_atoms)

    mnodes = _obj_names(len(qnodes))
    mpreds = _link_names(len(qpreds), 0)
    rng.shuffle(mnodes)
    rng.shuffle(mpreds)

    obj_inv = dict(zip(qnodes, mnodes))
    pred_inv = dict(zip(qpreds, mpreds))
    flip = {p: rng.random() < reverse_rate for p in qpreds}

    def inv(atom):
        p, a, b = atom
        q = pred_inv[p]
        return (q, obj_inv[b], obj_inv[a]) if flip[p] else (q, obj_inv[a], obj_inv[b])

    return edict(context={inv(x) for x in ctx}, consequence=inv(q_consequence))


def _random_case(n_nodes, n_preds, n_context, rng):
    nodes = _obj_names(n_nodes)
    preds = _link_names(n_preds, 0)
    ctx = _rand_atoms(nodes, preds, n_context, rng)
    if not ctx:
        return None
    cons = _candidate_consequences(ctx, nodes, preds, rng)
    if not cons:
        return None
    return edict(context=ctx, consequence=cons[0])


def _hard_negative_case(q_before, q_consequence, m, rng, reverse_rate=0.15):
    case = _inverse_case(q_before, q_consequence, m, rng, reverse_rate=reverse_rate)
    ctx = set(case.context)
    cons = tuple(case.consequence)  # edict coerces the consequence tuple to a list; restore hashability
    nodes, preds = _nodes(ctx | {cons}), _preds(ctx | {cons})
    for atom in rng.sample(list(ctx), len(ctx)):
        p, a, b = atom
        edits = [(p, b, a)]
        edits += [(p, x, b) for x in nodes if x not in (a, b)]
        edits += [(p, a, x) for x in nodes if x not in (a, b)]
        edits += [(q, a, b) for q in preds if q != p]
        rng.shuffle(edits)
        for new_atom in edits:
            if new_atom not in ctx:
                ctx.remove(atom)
                ctx.add(new_atom)
                case.context = ctx
                return case
    return None


def _transported_consequences(case, q_before, allow_reverse=True, injective_predicates=True, cap=16):
    c_atoms = set(case.context)
    c_cons = tuple(case.consequence)  # edict coerces the consequence tuple to a list; restore hashability
    q_atoms = set(q_before)

    cnodes = _nodes(c_atoms | {c_cons})
    qnodes = _nodes(q_atoms)
    cpreds = _preds(c_atoms | {c_cons})
    qpreds = _preds(q_atoms)

    if len(cnodes) > len(qnodes) or len(cpreds) > len(qpreds):
        return []

    flips = [False, True] if allow_reverse else [False]
    c_order = sorted(c_atoms, key=lambda atom: sum(atom[1:].count(x) for x in cnodes), reverse=True)
    out = set()

    def bind(m, used, a, x):
        if a in m:
            return m if m[a] == x else None
        if x in used:
            return None
        return {**m, a: x}

    def complete_nodes(omap, names):
        missing = [x for x in names if x not in omap]
        pool = [x for x in qnodes if x not in omap.values()]
        for vals in it.permutations(pool, len(missing)):
            yield {**omap, **dict(zip(missing, vals))}

    def search(i, omap, pmap, fmap):
        if len(out) >= cap:
            return
        if i == len(c_order):
            p, a, b = c_cons
            preds = [pmap[p]] if p in pmap else [q for q in qpreds if not injective_predicates or q not in pmap.values()]
            for q in preds:
                for flip in ([fmap[p]] if p in fmap else flips):
                    for omap2 in complete_nodes(omap, [a, b]):
                        x, y = omap2[a], omap2[b]
                        out.add((q, y, x) if flip else (q, x, y))
                        if len(out) >= cap:
                            return
            return

        p, a, b = c_order[i]
        for q, x, y in q_atoms:
            for flip in flips:
                if p in pmap and (pmap[p] != q or fmap[p] != flip):
                    continue
                if p not in pmap and injective_predicates and q in pmap.values():
                    continue
                x, y = (y, x) if flip else (x, y)
                omap2 = bind(omap, set(omap.values()), a, x)
                if omap2 is None:
                    continue
                omap2 = bind(omap2, set(omap2.values()), b, y)
                if omap2 is not None:
                    search(i + 1, omap2, {**pmap, p: q}, {**fmap, p: flip})

    search(0, {}, {}, {})
    return sorted(out)


def _all_consequences(cases, q_before):
    hits = defaultdict(list)
    for case in cases:
        for cons in _transported_consequences(case, q_before):
            hits[cons].append(case.id)
    return hits


def _canonical_case_structure(context, consequence):
    """Canonicalize a case up to entity/relation names and relation direction."""
    context = [tuple(atom) for atom in context]
    consequence = tuple(consequence)
    predicates = sorted({atom[0] for atom in context + [consequence]})

    def canonical_entities(records):
        entities = sorted({x for _, _, a, b in records for x in (a, b)})

        def refine(colors):
            while True:
                signatures = {}
                for entity in entities:
                    incident = []
                    for kind, predicate, left, right in records:
                        if entity == left:
                            incident.append(("out", kind, predicate, colors[right]))
                        if entity == right:
                            incident.append(("in", kind, predicate, colors[left]))
                    signatures[entity] = (colors[entity], tuple(sorted(incident)))
                palette = {value: i for i, value in enumerate(sorted(set(signatures.values()), key=repr))}
                updated = {entity: palette[signature] for entity, signature in signatures.items()}
                if updated == colors:
                    return colors
                colors = updated

        def search(colors):
            colors = refine(colors)
            cells = defaultdict(list)
            for entity, color in colors.items():
                cells[color].append(entity)
            ambiguous = [cell for _, cell in sorted(cells.items()) if len(cell) > 1]
            if not ambiguous:
                names = {entity: color for entity, color in colors.items()}
                return tuple(sorted((kind, predicate, names[left], names[right])
                                    for kind, predicate, left, right in records))
            cell = min(ambiguous, key=lambda values: (len(values), colors[values[0]]))
            return min(search({**colors, entity: max(colors.values()) + 1}) for entity in cell)

        return search({entity: 0 for entity in entities})

    candidates = []
    for order in it.permutations(predicates):
        predicate_ids = {predicate: i for i, predicate in enumerate(order)}
        for reversed_bits in it.product((False, True), repeat=len(predicates)):
            reverse = dict(zip(predicates, reversed_bits))
            records = []
            for kind, atoms in ((0, context), (1, [consequence])):
                for predicate, left, right in atoms:
                    if reverse[predicate]:
                        left, right = right, left
                    records.append((kind, predicate_ids[predicate], left, right))
            candidates.append(canonical_entities(records))
    return min(candidates)


@dataclass
class AnalogicalCaseMatchingConfig(Config):
    n_query_objects: int = 5
    n_query_links: int = 3
    n_query_facts: int = 6
    n_query_facts_range: tuple | None = None
    n_cases: int = 3
    n_gold_cases: int = 1
    context_facts: int = 4
    memory_distractors: int = 0
    memory_distractors_range: tuple | None = None
    index_answer_rate: float = 1.00
    no_match_prompt_rate: float = 0.10
    no_match_answer_rate: float = 0.50
    reverse_rate: float = 0.15
    max_attempts: int = 800

    def apply_difficulty(self, level):
        self.n_query_objects = sround(self.n_query_objects + level)
        self.n_query_facts = sround(self.n_query_facts + 2 * level)
        if self.n_query_facts_range is not None:
            lo, hi = self.n_query_facts_range
            self.n_query_facts_range = (sround(lo + 2 * level), sround(hi + 2 * level))
        self.n_cases = sround(self.n_cases + level)
        self.context_facts = sround(self.context_facts + 0.5 * level)
        self.reverse_rate = min(0.5, self.reverse_rate + 0.05 * level)


class AnalogicalCaseMatching(Task):
    summary = "Retrieve analogical cases matching query objects, links, and logical facts."
    def __init__(self, config=None):
        super().__init__(config=config or AnalogicalCaseMatchingConfig())

    def _choose_answer_format(self, rng):
        return "index" if rng.random() < self.config.index_answer_rate else "fact"

    def _format_answer(self, q_answer, gold_ids, answer_format):
        if answer_format == "index":
            return " ".join(gold_ids)
        return _compact(q_answer)

    def generate_entry(self):
        k = self.config
        rng = random

        n_obj = int(k.n_query_objects)
        n_rel = int(k.n_query_links)
        n_facts = _sample_param(k.n_query_facts, k.n_query_facts_range, rng)
        n_mem_noise = _sample_param(k.memory_distractors, k.memory_distractors_range, rng)
        n_cases = int(k.n_cases)
        allow_no_match = rng.random() < k.no_match_prompt_rate
        no_match = allow_no_match and rng.random() < k.no_match_answer_rate
        n_gold = 0 if no_match else int(k.n_gold_cases)
        n_ctx = int(k.context_facts)

        qnodes = _query_names(n_obj)
        qpreds = _link_names(n_rel, 2)

        for _ in range(int(k.max_attempts)):
            q_before = _rand_atoms(qnodes, qpreds, n_facts, rng)
            if not _weakly_connected(q_before):
                continue

            consequences = _candidate_consequences(q_before, qnodes, qpreds, rng)
            if not consequences:
                continue
            q_answer = consequences[0]

            cases = []
            for _ in range(n_gold):
                case = _inverse_case(
                    q_before,
                    q_answer,
                    n_ctx + n_mem_noise,
                    rng,
                    reverse_rate=k.reverse_rate,
                )
                cases.append(case)
            if len(cases) < n_gold:
                continue

            tries = 0
            while len(cases) < n_cases and tries < 800:
                tries += 1
                if rng.random() < 0.75:
                    case = _hard_negative_case(
                        q_before, q_answer, n_ctx + n_mem_noise, rng, reverse_rate=k.reverse_rate
                    )
                else:
                    case = _random_case(
                        n_nodes=min(n_obj, max(3, len(_nodes(q_before)))),
                        n_preds=n_rel,
                        n_context=n_ctx + n_mem_noise,
                        rng=rng,
                    )
                if case is None:
                    continue
                if not _transported_consequences(case, q_before, cap=1):
                    cases.append(case)

            if len(cases) < n_cases:
                continue

            rng.shuffle(cases)
            for i, case in enumerate(cases):
                case.id = f"M{i}"

            hits = _all_consequences(cases, q_before)
            expected = set() if no_match else {q_answer}
            if set(hits) != expected:
                continue

            gold_ids = [] if no_match else sorted(hits[q_answer], key=lambda x: int(x[1:]))
            answer_format = self._choose_answer_format(rng)
            answer = "None" if no_match else self._format_answer(q_answer, gold_ids, answer_format)

            md = edict(
                cases=[
                    edict(
                        id=case.id,
                        context=sorted(case.context),
                        consequence=case.consequence,
                    )
                    for case in cases
                ],
                query_context=sorted(q_before),
                answer_atom=q_answer,
                matching_case_ids=gold_ids,
                allow_no_match=allow_no_match,
                no_match=no_match,
                answer_format=answer_format,
                answer=answer,
                params=dict(
                    n_query_objects=n_obj,
                    n_query_links=n_rel,
                    n_query_facts=n_facts,
                    n_cases=n_cases,
                    n_gold_cases=n_gold,
                    context_facts=n_ctx,
                    memory_distractors=n_mem_noise,
                    index_answer_rate=k.index_answer_rate,
                    no_match_prompt_rate=k.no_match_prompt_rate,
                    no_match_answer_rate=k.no_match_answer_rate,
                    reverse_rate=k.reverse_rate,
                ),
            )
            return Entry(metadata=md, answer=answer)

        raise RuntimeError("generation budget exhausted")

    def _render_case(self, case, include_consequence):
        facts = ", ".join(_compact(atom) for atom in sorted(case["context"]))
        if include_consequence:
            facts += f" -> {_compact(case['consequence'])}"
        return f"{case['id']}: {facts}"

    def render_prompt(self, metadata):
        answer_format = metadata.get("answer_format", "fact")
        rule = "under consistent entity/relation renaming and per-relation direction reversal"
        if answer_format == "index":
            plural = int(self.config.n_gold_cases) > 1
            subject = "cases" if plural else "case"
            verb = "match" if plural else "matches"
            ids = "their IDs" if plural else "its ID"
            prompt = f"Which {subject} {verb} Query {rule}? Answer with {ids}"
            if metadata.get("allow_no_match"):
                prompt += ", or None."
            else:
                prompt += "."
        else:
            prompt = f"Infer Query's missing fact by mapping a case {rule}. Answer with one fact"
            if metadata.get("allow_no_match"):
                prompt += ", or None if no case matches."
            else:
                prompt += "."

        lines = [prompt, ""]
        for case in metadata["cases"]:
            lines.append(self._render_case(case, include_consequence=answer_format != "index"))

        query = ", ".join(_compact(atom) for atom in sorted(metadata["query_context"]))
        if answer_format != "index":
            query += " -> ?"
        lines.append(f"Query: {query}")

        return "\n".join(lines)

    def deduplication_key(self, problem):
        metadata = problem.metadata
        # Exact graph canonization has a rare factorial tail on large symmetric
        # query graphs. Preserve generation scalability and never risk a false
        # merge by falling back to the exact rendered prompt/answer pair.
        if len(metadata["query_context"]) > 16:
            return super().deduplication_key(problem)
        cases = sorted(
            _canonical_case_structure(case["context"], case["consequence"])
            for case in metadata["cases"]
        )
        query = _canonical_case_structure(
            metadata["query_context"], metadata["answer_atom"]
        )
        return (
            tuple(cases), query, metadata["answer_format"],
            metadata["allow_no_match"], metadata["no_match"],
        )

    def score_answer(self, answer, entry):
        text = str(answer).strip().casefold().rstrip(".")
        if entry.metadata.get("no_match"):
            return 1.0 if text == "none" else 0.0

        if entry.metadata.get("answer_format") == "index":
            pred_ids = re.findall(r"\bM\d+\b", str(answer))
            return 1.0 if pred_ids == list(entry.metadata["matching_case_ids"]) else 0.0

        gold = tuple(entry.metadata["answer_atom"])  # edict stores answer_atom as a list; parse yields a tuple
        pred = _parse_sent(answer)
        return 1.0 if pred is not None and tuple(pred) == gold else 0.0
