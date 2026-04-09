import itertools
import os
import random
import re
import sys
from dataclasses import dataclass

import owlready2
import owlready2 as owl
from reasoning_core.template import Task, Problem, Config, edict


LABELS = ("entailment", "contradiction", "neutral")


@dataclass
class KnowledgeReasoningConfig(Config):
    n_entities: int = 8
    n_facts: int = 12
    n_hypotheses: int = 6
    retries: int = 8

    def update(self, c=1):
        # Smooth curriculum; framework rounds int fields on read.
        self.n_entities += 0.5 * c
        self.n_facts += 1.5 * c
        self.n_hypotheses += 0.5 * c
        self.retries += 0.25 * c


def build_ontology(world: owlready2.World):
    onto = world.get_ontology("http://synth/onto.owl")
    with onto:
        class Entity(owlready2.Thing): pass
        class Person(Entity): pass
        class Item(Entity): pass
        class Location(Entity): pass
        class Organization(Entity): pass
        class Event(Entity): pass
        owlready2.AllDisjoint([Person, Item, Location, Organization, Event])

        class located_in(Entity >> Location, owlready2.TransitiveProperty): pass
        class contains(Location >> Entity, owlready2.TransitiveProperty):
            inverse_property = located_in

        class part_of(Item >> Item, owlready2.TransitiveProperty): pass
        class has_part(Item >> Item, owlready2.TransitiveProperty):
            inverse_property = part_of

        class before(Event >> Event, owlready2.TransitiveProperty): pass
        class after(Event >> Event, owlready2.TransitiveProperty):
            inverse_property = before

        class ancestor_of(Person >> Person, owlready2.TransitiveProperty): pass
        class descendant_of(Person >> Person, owlready2.TransitiveProperty):
            inverse_property = ancestor_of

        class adjacent_to(Location >> Location,
                          owlready2.SymmetricProperty, owlready2.IrreflexiveProperty): pass
        class sibling_of(Person >> Person,
                         owlready2.SymmetricProperty, owlready2.IrreflexiveProperty): pass
        class partnered_with(Organization >> Organization,
                             owlready2.SymmetricProperty, owlready2.IrreflexiveProperty): pass

        class works_for(Person >> Organization,
                        owlready2.FunctionalProperty, owlready2.IrreflexiveProperty): pass
        class employs(Organization >> Person,
                      owlready2.InverseFunctionalProperty, owlready2.IrreflexiveProperty):
            inverse_property = works_for

        class owns(Person >> Item): pass
        class owned_by(Item >> Person, owlready2.FunctionalProperty):
            inverse_property = owns

        class participates_in(Person >> Event): pass
        class has_participant(Event >> Person):
            inverse_property = participates_in

    return onto


def surface(name: str) -> str:
    return name.replace("_", " ")


def render_entity(ent) -> str:
    return surface(ent.name)


def silent_reason(world):
    with open(os.devnull, "w") as nul:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = nul
        try:
            owlready2.sync_reasoner(world, infer_property_values=True)
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def named(xs):
    return [x for x in xs if hasattr(x, "ancestors")]


def match(ind, classes):
    cs = named(classes)
    return not cs or any(isinstance(ind, c) for c in cs)


def anc(c):
    try:
        return [x for x in c.ancestors() if hasattr(x, "name")]
    except Exception:
        return [c] if hasattr(c, "name") else []


def disjoint_pairs(onto):
    out = set()
    for c in onto.classes():
        for dj in c.disjoints():
            xs = [x for x in dj.entities if hasattr(x, "ancestors")]
            out |= set(itertools.permutations(xs, 2))
    return out


def class_disjoint(a, b, dpairs):
    return any((x, y) in dpairs for x in anc(a) for y in anc(b))


def has_edge(u, P, v):
    val = getattr(u, P.name)
    return val == v if issubclass(P, owlready2.FunctionalProperty) else v in val


def add_edge(u, P, v):
    if issubclass(P, owlready2.FunctionalProperty):
        setattr(u, P.name, v)
    else:
        getattr(u, P.name).append(v)


def reachable(graph, src, dst):
    stack, seen = [src], {src}
    while stack:
        u = stack.pop()
        for v in graph.get(u, ()):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def violates(u, P, v, ents, dpairs):
    if u is v and issubclass(P, owlready2.IrreflexiveProperty):
        return True
    if issubclass(P, owlready2.AsymmetricProperty) and has_edge(v, P, u):
        return True
    if issubclass(P, owlready2.FunctionalProperty):
        cur = getattr(u, P.name)
        if cur is not None and cur != v:
            return True
    if issubclass(P, owlready2.InverseFunctionalProperty):
        if any(x is not u and has_edge(x, P, v) for x in ents):
            return True
    for obj, cls in ((u, P.domain), (v, P.range)):
        cs = named(cls)
        if cs and not match(obj, cs) and all(class_disjoint(obj.__class__, c, dpairs) for c in cs):
            return True
    return False


def pair_ok(u, P, v, ents, dpairs, graphs):
    if u is v or has_edge(u, P, v):
        return False
    if not match(u, P.domain) or not match(v, P.range):
        return False
    if violates(u, P, v, ents, dpairs):
        return False
    if issubclass(P, owlready2.SymmetricProperty) and has_edge(v, P, u):
        return False
    if issubclass(P, owlready2.TransitiveProperty):
        g = graphs.get(P, {})
        if reachable(g, u, v) or reachable(g, v, u):
            return False
    return True


def pick_pair(P, ents, dpairs, graphs, tries=100):
    for _ in range(tries):
        u, v = random.sample(ents, 2)
        if pair_ok(u, P, v, ents, dpairs, graphs):
            return u, v
    return None


def pick_chain(P, ents, dpairs, graphs, tries=150):
    if len(ents) < 3:
        return None
    g = graphs.get(P, {})
    for _ in range(tries):
        a, b, c = random.sample(ents, 3)
        if not pair_ok(a, P, b, ents, dpairs, graphs):
            continue
        g.setdefault(a, set()).add(b)
        ok = pair_ok(b, P, c, ents, dpairs, graphs)
        g[a].discard(b)
        if not g[a]:
            g.pop(a, None)
        if ok:
            return a, b, c
    return None


def pick_one_per_inverse(props):
    out, seen = [], set()
    for P in props:
        if P in seen:
            continue
        Q = getattr(P, "inverse_property", None)
        if Q in props and Q not in seen:
            out.append(random.choice([P, Q]))
            seen |= {P, Q}
        else:
            out.append(P)
            seen.add(P)
    return out


class KnowledgeReasoning(Task):
    def __init__(self, config=None):
        super().__init__(config=config or KnowledgeReasoningConfig())

    def _build_world(self):
        w = owlready2.World()
        return w, build_ontology(w)

    def _generate_once(self):
        w, onto = self._build_world()
        try:
            return self._do_generate(w, onto, self.config)
        finally:
            w.close()

    def _do_generate(self, world, onto, cfg):
        n_entities = cfg.n_entities
        n_facts = cfg.n_facts
        n_hypotheses = cfg.n_hypotheses

        leaves = [c for c in onto.classes()
                  if c is not owlready2.Nothing and not list(c.subclasses())]
        all_props = list(onto.object_properties())
        props = pick_one_per_inverse(all_props)
        if not leaves or not all_props:
            raise RuntimeError("empty ontology")

        pool = random.sample(leaves, len(leaves))
        ents = [pool[i % len(pool)](f"{pool[i % len(pool)].name}_{i}")
                for i in range(n_entities)]
        owlready2.AllDifferent(ents)

        dpairs = disjoint_pairs(onto)
        facts = [f"{render_entity(e)} is a {surface(e.__class__.name)}." for e in ents]
        explicit = set()
        graphs = {P: {} for P in props if issubclass(P, owlready2.TransitiveProperty)}

        def add(u, P, v):
            add_edge(u, P, v)
            explicit.add((u.name, P.name, v.name))
            facts.append(f"{render_entity(u)} {surface(P.name)} {render_entity(v)}.")
            if P in graphs:
                graphs[P].setdefault(u, set()).add(v)

        added = 0

        # 1) transitive chains first: strongest reasoning signal
        for P in random.sample(
            [p for p in props if issubclass(p, owlready2.TransitiveProperty)],
            k=min(4, sum(issubclass(p, owlready2.TransitiveProperty) for p in props))
        ):
            if added + 2 > n_facts:
                break
            ch = pick_chain(P, ents, dpairs, graphs)
            if ch:
                add(ch[0], P, ch[1])
                add(ch[1], P, ch[2])
                added += 2

        # 2) symmetric / inverse facts
        for P in props:
            if added >= n_facts:
                break
            if issubclass(P, owlready2.SymmetricProperty) or getattr(P, "inverse_property", None):
                pair = pick_pair(P, ents, dpairs, graphs)
                if pair:
                    add(pair[0], P, pair[1])
                    added += 1

        # 3) fill the rest
        for _ in range(n_facts * 60):
            if added >= n_facts:
                break
            P = random.choice(props)
            pair = pick_pair(P, ents, dpairs, graphs, tries=15)
            if pair:
                add(pair[0], P, pair[1])
                added += 1

        if added < max(3, n_facts // 2):
            raise RuntimeError("too few facts")

        silent_reason(world)
        if list(world.inconsistent_classes()):
            raise RuntimeError("ontology inconsistent")

        pools = {k: [] for k in LABELS}
        seen = set()

        for u, P, v in itertools.product(ents, all_props, ents):
            if u is v:
                continue
            key = (u.name, P.name, v.name)
            if key in seen:
                continue
            seen.add(key)

            txt = f"{u.name} {surface(P.name)} {v.name}"
            if has_edge(u, P, v):
                inv = getattr(P, "inverse_property", None)
                is_trivial = (
                    key in explicit
                    or (issubclass(P, owlready2.SymmetricProperty) and (v.name, P.name, u.name) in explicit)
                    or (inv and (v.name, inv.name, u.name) in explicit)
                )
                if not is_trivial:
                    infer_score = (
                        2 * issubclass(P, owlready2.TransitiveProperty)
                        + 1 * issubclass(P, owlready2.SymmetricProperty)
                        + 1 * bool(inv)
                    )
                    pools["entailment"].append((infer_score, txt))
            elif violates(u, P, v, ents, dpairs) and match(u, P.domain) and match(v, P.range):
                pools["contradiction"].append(txt)
            elif match(u, P.domain) and match(v, P.range):
                pools["neutral"].append(txt)

        pools["entailment"].sort(key=lambda x: (-x[0], x[1]))
        pools["entailment"] = [t for _, t in pools["entailment"]]
        random.shuffle(pools["contradiction"])
        random.shuffle(pools["neutral"])

        if sum(len(v) for v in pools.values()) < n_hypotheses:
            raise RuntimeError("too few hypotheses")

        chosen = []
        per = max(1, n_hypotheses // 3)

        for lbl in LABELS:
            chosen += [(h, lbl) for h in pools[lbl][:per]]

        if len(chosen) < n_hypotheses:
            leftovers = []
            for lbl in LABELS:
                leftovers += [(h, lbl) for h in pools[lbl][per:]]
            random.shuffle(leftovers)
            chosen += leftovers[:n_hypotheses - len(chosen)]

        if len(chosen) < n_hypotheses:
            raise RuntimeError("not enough hypotheses after balancing")

        random.shuffle(chosen)
        random.shuffle(facts)

        hyps = [h.replace("_", " ") for h, _ in chosen]
        labels = [l for _, l in chosen]
        ans = "\n".join(f"{i+1}. {l}" for i, l in enumerate(labels))

        return Problem(
            metadata=edict(facts=facts, hypotheses=hyps, labels=labels),
            answer=ans,
        )

    def generate(self):
        for _ in range(int(self.config.retries)):
            try:
                return self._generate_once()
            except RuntimeError:
                pass
        raise RuntimeError("Failed to generate after retries")

    def prompt(self, metadata):
        facts = "\n".join(f"- {f}" for f in metadata.facts)
        hyps = "\n".join(f"{i+1}. {h}" for i, h in enumerate(metadata.hypotheses))
        return (
            f"Given the following facts:\n{facts}\n\n"
            "For each hypothesis, predict: entailment, contradiction, or neutral.\n"
            "Format exactly as a numbered list (e.g. '1. entailment').\n\n"
            f"Hypotheses:\n{hyps}\n\n"
            "Answer only."
        )

    def score_answer(self, answer, entry):
        rx = re.compile(r"^\s*(\d+)\s*[.)]\s*(entailment|contradiction|neutral)\s*$", re.I)

        def parse(text):
            out = []
            for line in text.strip().splitlines():
                m = rx.match(line)
                if m:
                    out.append((int(m.group(1)), m.group(2).lower()))
            return out

        exp, pred = parse(entry.answer), parse(answer)
        if not exp or not pred or len(exp) != len(pred):
            return 0.0

        nums = [i for i, _ in exp]
        if nums != list(range(1, len(exp) + 1)):
            return 0.0
        if [i for i, _ in pred] != nums:
            return 0.0

        return sum(e == p for (_, e), (_, p) in zip(exp, pred)) / len(exp)

