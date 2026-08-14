import json
import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


_NAMES = ("Ari", "Bela", "Cleo", "Dara", "Enzo", "Fara", "Gio", "Hana", "Ivo", "Juna", "Kian", "Lumi")
_RELATIONS = {
    "supports": (
        ("{s} supports {o}.", "{s} does not support {o}."),
        ("{o} receives support from {s}.", "{o} does not receive support from {s}."),
    ),
    "manages": (
        ("{s} manages {o}.", "{s} does not manage {o}."),
        ("{o} reports to {s}.", "{o} does not report to {s}."),
    ),
    "precedes": (
        ("{s} precedes {o}.", "{s} does not precede {o}."),
        ("{o} follows {s}.", "{o} does not follow {s}."),
    ),
    "visits": (
        ("{s} visits {o}.", "{s} does not visit {o}."),
        ("{o} is visited by {s}.", "{o} is not visited by {s}."),
    ),
}
_TARGET_RELATIONS = ("supports", "manages", "precedes")
_PROPERTIES = ("calm", "careful", "direct", "eager", "focused", "gentle", "patient", "quiet", "steady", "vivid")


def _json_score(answer, entry):
    try:
        predicted = json.loads(str(answer).strip())
        reference = json.loads(entry["answer"] if isinstance(entry, dict) else entry.answer)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return float(predicted == reference)


def _json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


@dataclass
class TypedRelationExtractionConfig(Config):
    n_entities: int = 5
    n_sentences: int = 7
    max_attempts: int = 100

    def apply_difficulty(self, level):
        self.n_entities = sround(self.n_entities + 0.5 * level)
        self.n_sentences = sround(self.n_sentences + 1.4 * level)
        self.max_attempts = sround(self.max_attempts + 10 * level)


class TypedRelationExtraction(GeneratedMixin, Task):
    summary = "Extract the complete set of typed relations with sentence provenance while ignoring negated and irrelevant statements."
    config_cls = TypedRelationExtractionConfig

    def generate_entry(self):
        cfg = self.config
        names = random.sample(_NAMES, min(cfg.n_entities, len(_NAMES)))
        for _ in range(cfg.max_attempts):
            sentences, records, seen = [], [], set()
            for i in range(1, cfg.n_sentences + 1):
                relation = random.choice(tuple(_RELATIONS))
                s, o = random.sample(names, 2)
                key = relation, s, o
                if key in seen:
                    continue
                seen.add(key)
                positive = random.random() >= 0.28
                pattern = random.choice(_RELATIONS[relation])[0 if positive else 1]
                sentences.append(pattern.format(s=s, o=o))
                if positive and relation in _TARGET_RELATIONS:
                    records.append({"relation": relation, "source": s, "target": o, "evidence": len(sentences)})
            if len(sentences) < max(4, cfg.n_sentences // 2) or len(records) < 2 or len(records) == len(sentences):
                continue
            metadata = edict(sentences=sentences, records=records)
            return Entry(metadata=metadata, answer=_json(records))
        raise RuntimeError("Failed to generate a varied relation-extraction instance")

    def render_prompt(self, m):
        text = "\n".join(f"{i}. {s}" for i, s in enumerate(m.sentences, 1))
        return (
            f"Statements:\n{text}\n\n"
            "Extract every affirmative supports, manages, and precedes relation. Interpret reversed wording semantically. "
            "Ignore negated statements and all other relation types. The answer is a JSON array in evidence-sentence order. "
            "Each object has exactly the keys relation, source, target, evidence, where evidence is the sentence number."
        )

    def score_answer(self, answer, entry):
        return _json_score(answer, entry)

    def balancing_key(self, problem):
        return min(5, len(problem.metadata.records))


@dataclass
class EvidenceSufficiencyConfig(Config):
    n_entities: int = 4
    n_properties: int = 6
    n_evidence: int = 8
    claim_atoms: int = 2

    def apply_difficulty(self, level):
        self.n_entities = sround(self.n_entities + 0.35 * level)
        self.n_properties = sround(self.n_properties + 0.5 * level)
        self.n_evidence = sround(self.n_evidence + 1.3 * level)
        self.claim_atoms = sround(self.claim_atoms + 0.3 * level)


class EvidenceSufficiency(GeneratedMixin, Task):
    summary = "Distinguish sufficient, contradictory, and merely related evidence and identify the exact witness sentences."
    config_cls = EvidenceSufficiencyConfig

    def generate_entry(self):
        cfg = self.config
        names = random.sample(_NAMES, min(cfg.n_entities, len(_NAMES)))
        properties = random.sample(_PROPERTIES, min(cfg.n_properties, len(_PROPERTIES)))
        target = random.choice(names)
        claim = random.sample(properties, min(cfg.claim_atoms, len(properties)))
        verdict = random.choice(("supported", "contradicted", "insufficient"))
        facts = []

        if verdict == "supported":
            facts.extend((target, p, True) for p in claim)
        elif verdict == "contradicted":
            neg = random.choice(claim)
            facts.append((target, neg, False))
            facts.extend((target, p, True) for p in claim if p != neg and random.random() < 0.6)
        else:
            shown = random.sample(claim, random.randrange(1, len(claim))) if len(claim) > 1 else []
            facts.extend((target, p, True) for p in shown)

        forbidden = {(target, p) for p in claim}
        pool = [(n, p, sign) for n in names for p in properties for sign in (True, False)
                if (n, p) not in forbidden]
        random.shuffle(pool)
        for item in pool:
            if len(facts) >= cfg.n_evidence:
                break
            if item[:2] not in {(n, p) for n, p, _ in facts}:
                facts.append(item)
        random.shuffle(facts)

        sentences, positive_ids, negative_ids = [], {}, {}
        for i, (name, prop, sign) in enumerate(facts, 1):
            if sign:
                text = random.choice((f"{name} is {prop}.", f"The evidence states that {name} is {prop}."))
                positive_ids[(name, prop)] = i
            else:
                text = random.choice((f"{name} is not {prop}.", f"The evidence states that {name} is not {prop}."))
                negative_ids[(name, prop)] = i
            sentences.append(text)

        if verdict == "supported":
            witness = sorted(positive_ids[(target, p)] for p in claim)
        elif verdict == "contradicted":
            witness = [min(negative_ids[(target, p)] for p in claim if (target, p) in negative_ids)]
        else:
            witness = []
        answer = {"verdict": verdict, "evidence": witness}
        return Entry(metadata=edict(sentences=sentences, target=target, claim=claim, verdict=verdict), answer=_json(answer))

    def render_prompt(self, m):
        evidence = "\n".join(f"{i}. {s}" for i, s in enumerate(m.sentences, 1))
        claim = " and ".join(f"{m.target} is {p}" for p in m.claim)
        return (
            f"Evidence:\n{evidence}\n\nClaim: {claim}.\n"
            "Use only the supplied evidence. The verdict is supported iff every conjunct is explicitly affirmed; "
            "contradicted iff at least one conjunct is explicitly negated; otherwise it is insufficient. "
            "The answer is JSON with exactly keys verdict and evidence. For supported, evidence lists every sentence needed for the conjuncts; "
            "for contradicted, it lists the single smallest-numbered contradicting sentence; for insufficient, it is []."
        )

    def score_answer(self, answer, entry):
        return _json_score(answer, entry)

    def balancing_key(self, problem):
        return problem.metadata.verdict
