from gramforge import Substitution, Constraint, generate, init_grammar
S,C=Substitution,Constraint

import funcy as fc
from tqdm.auto import tqdm
import random, re, exrex
import itertools
from gramforge.solver_utils.tptp import split_clauses, run as _run, to_tptp, extract_inferences_and_formulas



AXIOMS = "fof(anywhere_ax, axiom, ![X]:anywhere(X))."
# in the grammar anywhere is treated as a location, we quantify "anyone anywhere or anyone in the room. 
# This axiom is necessary to say that people match being anywhere" 

def run(expr, **kw):
    return _run(AXIOMS + "\n" + expr, **kw)

from gramforge.assets import fol_nli_verbalization

import sys
from reasoning_core.template import Task, DevTask, Entry, Config, render_payload
from reasoning_core.utils import parse_space_ints
from gramforge.grammars.FOL import FOL_grammar
from easydict import EasyDict as edict
from functools import cache
from dataclasses import dataclass, field
from faker import Faker

import re

from ._logic_utils import cat_premises, satify_premise


eng, tptp = "eng","tptp"

ADJECTIVES = ['rich', 'quiet', 'old', 'tall', 'kind', 'brave', 'wise',
              'happy', 'strong', 'curious', 'patient', 'funny', 'generous', 'humble']

NAMES = ['mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy']
FEMALE_NAMES = ['mary', 'alice', 'susan', 'lucy', 'jane', 'anna', 'emma']
MALE_NAMES = ['paul', 'fred', 'john', 'michael', 'david', 'james', 'robert']
fake = Faker("en_US")

G = FOL_grammar

def safe_name(name):
    name = re.sub(r'[^a-z0-9_]+', '_', name.lower()).strip('_')
    return name if re.match(r'^[a-z]\w*$', name) else ''

def gendered_names(n, pronoun):
    if pronoun == "she":
        first_name = fake.first_name_female
    elif pronoun == "he":
        first_name = fake.first_name_male
    else:
        first_name = fake.first_name

    names = []
    seen = set()
    for _ in range(n * 20):
        name = safe_name(first_name())
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) >= n:
            return names
    fallback = {"she": FEMALE_NAMES, "he": MALE_NAMES}.get(pronoun, NAMES)
    return (names + fallback)[:n]

def make_hyps(G, N=1000):
    hyps = [generate(G().get_rules('hypothesis')[0], mode="sequential") for _ in range(N)]
    def dedup_by(xs, key):
        seen = set()
        return [x for x in xs if (k := key(x)) not in seen and not seen.add(k)]
    hyps=dedup_by(hyps, lambda x:x@eng)
    hyps_weights = [1 / (1+(h@tptp).count(')')**3) for h in hyps]
    return hyps, hyps_weights

def sample_hyps(hyps, hyps_weights, k=2000):
    return random.choices(hyps, weights=hyps_weights,k=k)


def generate_N_premises(n, G, mode="sequential"):
    empty_room = random.choice([True, False])
    gen = lambda n: generate(G(n, empty_room=empty_room), mode=mode)
    if n<=16:
        while True:
            x=gen(n)
            if valid(x):
                return x

    first_size = n % 16 or 16
    remaining_n = n - first_size

    x=gen(first_size)
    for _ in range(remaining_n // 16):
        x=satify_premise(cat_premises(x, gen(16)))

    return x

preds_pattern = list(exrex.generate('pred[a-z]'))
npreds_pattern = list(exrex.generate('~pred[a-z]'))
prop_pattern = list(exrex.generate('proposition[a-z]'))
nprop_pattern = list(exrex.generate('~proposition[a-z]'))


def predicate_mapping(seed):
    rng = random.Random(seed)
    source = sorted(list(fol_nli_verbalization.predicates))
    preds = rng.sample(source, len(preds_pattern))
    npreds = [fol_nli_verbalization.negate_predicate(p) for p in preds]
    prop_pairs = list(zip(fol_nli_verbalization.short_propositions,
                          fol_nli_verbalization.neg_short_propositions))
    rng.shuffle(prop_pairs)
    props, nprops = zip(*prop_pairs)
    return {**dict(zip(npreds_pattern, npreds)),
            **dict(zip(preds_pattern, preds)),
            **dict(zip(nprop_pattern, nprops)),
            **dict(zip(prop_pattern, props))}

def verbalize_predicates(x, seed=None, strip_underscores=True):  # now thin wrapper
    mapping = predicate_mapping(seed)
    for k in sorted(mapping, key=len, reverse=True):
        v = mapping[k] if strip_underscores else mapping[k].replace(' ', '_')
        x = x.replace(k, v)
    return x.replace('_', ' ') if strip_underscores else x

    
def valid(x):
    for p in "", "~":
        status= run(f"fof(f,axiom,{p}({x@tptp})).").status
        assert  status in ["Satisfiable", "Unsatisfiable", "Refutation not found", "Time limit"]
        if status!="Satisfiable":
            return False
    return True


@dataclass
class LogicConfig(Config):
    n_formulas: int = 6
    generation_algorithm: str = "sequential"
    n_names: int = 3
    n_adjectives: int = 3
    pronouns: list = field(default_factory=lambda: ["she",'he'])
    bloat_skip_rate:float = 0.90
    def apply_difficulty(self, level):
        self.n_formulas *= 2 ** level
        self.n_names += level
        self.n_adjectives += level

def get_cot(text):
    lines, memo = [], {}
    for line in text.splitlines():
        m = re.match(r'^(\d+)\.\s+(.*)\s+\[(.*?)\]$', line)
        if not m:
            continue
        oid, form, meta = m.groups()

        rule, *rest = meta.split(maxsplit=1)
        ids = re.findall(r'\b(\d+)\b', meta)
        parents = [str(memo[i][0]) for i in ids if i in memo and rule != 'input']

        if len(parents) == 1 and ids[0] in memo and memo[ids[0]][1] == form:
            memo[oid] = memo[ids[0]]
            continue

        if rule == 'input':
            input_id = re.fullmatch(r'p?(\d+)', rest[0]) if rest else None
            ctx = 'H' if 'hyp' in meta or not input_id else f'P{input_id.group(1)}'
        elif rule == 'cnf' and not ids:
            ctx = 'H'
        else:
            ctx = f"{rule} {','.join(parents)}".rstrip()

        memo[oid] = (len(lines) + 1, form)
        lines.append(f"{len(lines) + 1}. [{ctx}] {form}")

    return "\n".join(lines)

def is_bloat(meta, label):
    rules = tuple(meta.proof['rules']) if meta.proof else ()
    bloat_signatures = {
        ('input', 'input', 'cnf', 'cnf', 'subsumption'),                     
        ('input', 'input', 'pure', 'cnf', 'cnf', 'subsumption'),             
        ('input', 'input', 'pure', 'pure', 'cnf', 'cnf', 'subsumption')      
    }
    return rules in bloat_signatures

class LogicNLI(DevTask):
    summary = "First-order logic natural language inference via automated theorem proving."

    def __init__(self, config=None):
        super().__init__(config=config or LogicConfig())
        self.pronoun = random.choice(self.config.pronouns)
        self.names = gendered_names(self.config.n_names, self.pronoun)
        self.adjectives = ADJECTIVES[:self.config.n_adjectives]
        G_hyp = fc.partial(FOL_grammar, names=self.names, adjs=self.adjectives,
                           include_propositional=True, empty_room=False,
                           pronoun=self.pronoun)
        self.hyps, self.hyps_weights=make_hyps(G_hyp)
        self.balancing_key_ratio=1/3

    def generate_entry(self):
        include_propositional = random.choice([True, False])
        empty_room = random.choice([True, False])
        self.G = fc.partial(FOL_grammar, names=self.names, adjs=self.adjectives,
                            empty_room=empty_room,
                            include_propositional=include_propositional,
                            pronoun=self.pronoun)
        meta = edict()
        for _ in range(100):    
            # generate premise
            x = generate_N_premises(self.config.n_formulas, self.G, mode=self.config.generation_algorithm)
            premise = split_clauses(x@tptp)

            # generate hypothesis
            xl = (x@eng).splitlines()
            for hyp in sample_hyps(self.hyps, self.hyps_weights):
                concepts = [x for x in re.findall(r'\w+(?=\()', hyp@tptp)  if x!='room']
                concept_match =  any(c in premise for c in concepts)
                if hyp@eng not in xl and valid(hyp) and concept_match :
                    break

            #compute label        
            proofs = [run(premise+f"\nfof(hyp,axiom,{prefix}({hyp@tptp})).")
                    for prefix in ("", "~")]
            meta.verbalize_seed = random.randint(0, int(1e6))
            meta.proof = proof = ([x for x in proofs if x.status=="Unsatisfiable"]+[None])[0]
            meta.cot = verbalize_predicates(get_cot(proof.proof), seed=meta.verbalize_seed, strip_underscores=False) if proof else ""
            labels = tuple([x.status for x in proofs])

            label = {
                ('Satisfiable', 'Unsatisfiable'): 'entailment',
                ('Satisfiable', 'Satisfiable'): 'neutral',
                ('Unsatisfiable', 'Satisfiable'): 'contradiction',
                ('Unsatisfiable', 'Unsatisfiable'): 'paradox'
            }.get(labels,'other')

            if label=="paradox":
                continue
            if label=="other":
                print("WARNING", "\n".join(str(p) for p in proofs))
                continue

            if is_bloat(meta, label):
                if random.random()<self.config.bloat_skip_rate:
                    continue
            
            meta.prem, meta.hyp = x.dict(), hyp.dict()
            meta.label = label
            meta.payload = {
                "premise": verbalize_predicates(meta.prem.eng, seed=meta.verbalize_seed),
                "hypothesis": verbalize_predicates(meta.hyp.eng, seed=meta.verbalize_seed),
            }
            mapping = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}
            return Entry(meta, mapping[label])

    def render_prompt(self, meta):
        return (
            f"{render_payload(meta.payload)}\n\n"
            "Is the hypothesis true given the premise? "
            "The answer is Yes, No, or Maybe."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().lower() == str(entry.answer).strip().lower())

    def balancing_key(self, problem):
        return problem.answer




@dataclass
class EvidenceRetrievalConfig(LogicConfig):
    bloat_skip_rate: float= 0.2

class EvidenceRetrieval(DevTask):
    summary = "Identify minimal necessary premises from logic theories to prove a hypothesis."
    def __init__(self, config=None):
        super().__init__(config=config or EvidenceRetrievalConfig())
        self.nli = LogicNLI(config=self.config)

    @staticmethod
    def compute_necessity(x):
        proof = x.metadata.get('proof')
        if not proof or not proof.input or not proof.indices:
            return False

        proof_lines = proof.input.splitlines()
        changes = dict()    
        for prefix in [f"fof({i}" for i in proof.indices]:
            ablation = [p for p in proof_lines if not p.startswith(prefix)] 
            y=run("\n".join(ablation))
            changes[prefix]=y.status
        return set(changes.values())=={"Satisfiable"}

    def generate_entry(self):
        for _ in range(200):
            self.nli.config = self.config
            x = self.nli.generate_entry()
            label = x.metadata.get('label', x.answer)
            label = {'yes': 'entailment', 'no': 'contradiction', 'maybe': 'neutral'}.get(str(label).lower(), label)
            proof = x.metadata.get('proof')
            answer = [i for i in proof.indices if i != 'hyp'] if proof else []
            if label in ('entailment', 'contradiction') and answer and self.compute_necessity(x):
                answer = ' '.join(f'{i}' for i in answer)
                return Entry(x.metadata, answer)

        raise RuntimeError("failed to generate evidence retrieval example with necessary proof inputs")

    def render_prompt(self, meta):
        prem_lines = [f"[{i}] {line}" for i, line in enumerate(meta.prem.eng.splitlines())]
        prem = '\n'.join(prem_lines)
        hyp = meta.hyp.eng
        verb = {'entailment':'entail','contradiction':'contradict'}.get(meta.label)
        P = (
            f"Premise:\n{prem}\n"
            f"Hypothesis:\n{hyp}\n\n"
            f"Which statements in the premise {verb} the hypothesis?\n"
            "Answer with space-separated indexes."
        )
        P=verbalize_predicates(P, seed=meta.verbalize_seed)
        return P
    
    def score_answer(self, answer, entry):
        reference = entry['answer']
        reference, answer = parse_space_ints(reference), parse_space_ints(answer)
        if not answer or reference is None:
            return 0.0
        reference, answer = set(reference), set(answer)
        return len(answer & reference) / len(answer | reference)

    def balancing_key(self, problem):
        return None
        #return len(problem.metadata.proof.indices) # too slow



@dataclass
class LogicFormalizationConfig(LogicConfig):
    n_formulas: int = 3

def _semanticize_tptp(expr, seed):
    used, out = {}, expr
    for k, v in sorted(predicate_mapping(seed).items(), key=lambda kv: len(kv[0]), reverse=True):
        if k.startswith('~'): continue
        name = safe_name(v.replace('not ', 'non_')) or k
        name += f"_{len(used)}" if name in used.values() else ""
        used[k] = name
        out = re.sub(rf'\b{k}\b', name, out)
    return out

def _equivalent(a, b):
    status = run(f"fof(eq, axiom, ~(({a}) <=> ({b}))).").status
    if status not in ("Satisfiable", "Unsatisfiable"):
        raise RuntimeError(f"Vampire inconclusive: {status}")
    return status == "Unsatisfiable"

def _alter_formula(f):
    def rename_content_pred(x):
        protected = {'in_the_room', 'anywhere', 'person', 'like', *ADJECTIVES}
        for m in re.finditer(r'\b([a-z]\w*)\(', x):
            if m.group(1) not in protected:
                return x[:m.start()] + f"changed_{m.group(1)}(" + x[m.end():]
        return f"(({x})&fresh_condition(fresh_object))"
    ops = [
        ("double_negation", lambda x: f"~(~({x}))"),
        ("idempotent_and", lambda x: f"(({x})&({x}))"),
        ("idempotent_or", lambda x: f"(({x})|({x}))"),
        ("negate_whole", lambda x: f"~({x})"),
        ("add_fresh_requirement", lambda x: f"(({x})&fresh_condition(fresh_object))"),
        ("add_fresh_alternative", lambda x: f"(({x})|fresh_condition(fresh_object))"),
        ("rename_one_predicate", rename_content_pred),
    ]
    random.shuffle(ops)
    return ops[0][0], ops[0][1](f)

class LogicFormalization(DevTask):
    summary = "Translate natural language premises into formal first-order logic formulas."
    def __init__(self, config=None):
        super().__init__(config=config or LogicFormalizationConfig())
        self.balancing_key_ratio = 0.5
        self.pronoun = random.choice(self.config.pronouns)
        self.names = gendered_names(self.config.n_names, self.pronoun)
        self.adjectives = ADJECTIVES[:self.config.n_adjectives]

    def generate_entry(self):
        include_propositional = random.choice([True, False])
        empty_room = False
        G = fc.partial(FOL_grammar, names=self.names, adjs=self.adjectives,
                       empty_room=empty_room,
                       include_propositional=include_propositional,
                       include_setup=False,
                       pronoun=self.pronoun)
        for _ in range(50):
            x = generate_N_premises(self.config.n_formulas, G,
                                    mode=self.config.generation_algorithm)
            seed = random.randint(0, int(1e6))
            gold = _semanticize_tptp((x@tptp).replace('room(', 'in_the_room('), seed)
            op, candidate = _alter_formula(gold)
            try:
                same = _equivalent(gold, candidate)
            except Exception:
                continue
            prem = x.dict()
            eng = verbalize_predicates(prem.eng, seed=seed)
            meta = edict(prem=prem, verbalize_seed=seed, candidate=candidate,
                         gold=gold, transformation=op,
                         payload={"English": eng, "TPTP": candidate})
            return Entry(meta, str(same))
        raise RuntimeError("Could not generate a Vampire-checkable contrastive formalization")

    def render_prompt(self, meta):
        meta = edict(meta)
        return (
            f"{render_payload(meta.payload)}\n\n"
            "Does the TPTP denotation match the English? "
            "The answer is True or False."
        )

    def score_answer(self, answer, entry):
        return float(answer.strip().lower() == entry.answer.lower())

    def balancing_key(self, problem):
        return problem.answer
