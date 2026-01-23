from unigram import Substitution, Constraint, generate, init_grammar
S,C=Substitution,Constraint

import funcy as fc
from tqdm.auto import tqdm
import random, re, exrex
import itertools
from unigram.solver_utils.tptp import split_clauses, run, to_tptp, extract_inferences_and_formulas
from unigram.assets import fol_nli_verbalization

import sys
from reasoning_core.template import Task, Problem, Config, register_dataset
from unigram.grammars.FOL import FOL_grammar
from easydict import EasyDict as edict
from tqdm.auto import tqdm
from functools import cache
from dataclasses import dataclass

import re

from ._logic_utils import cat_premises, satify_premise


eng, tptp = "eng","tptp"

ADJECTIVES = ['rich', 'quiet', 'old', 'tall', 'kind', 'brave', 'wise',
              'happy', 'strong', 'curious', 'patient', 'funny', 'generous', 'humble']

NAMES = ['mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy']

G = FOL_grammar

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
    gen = lambda n: generate(G(n), mode=mode)
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


def verbalize_predicates(x, seed=None, strip_underscores=True):
    rng = random.Random(seed)
    source = sorted(list(fol_nli_verbalization.predicates))
    preds = rng.sample(source, len(preds_pattern))
    
    mapping = {**dict(zip(npreds_pattern, [fol_nli_verbalization.negate_predicate(p) for p in preds])), 
               **dict(zip(preds_pattern, preds))}
    
    for k in sorted(mapping, key=len, reverse=True):
        v = mapping[k].replace(' ', '_') if not strip_underscores else mapping[k]
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
    n_names: int = 2
    n_adjectives: int = 2

    def update(self, c):
        self.n_formulas *= (1 + c)
        self.n_names += c
        self.n_adjectives += c
        
def get_cot(text: str) -> str:
    lines, memo = [], {}
    for line in text.splitlines():
        # Fix: Greedy match (.*) anchored ($) ensures we parse until the final metadata block
        if not (m := re.match(r'^(\d+)\.\s+(.*)\s+\[(.*?)\]$', line)): continue
        oid, form, meta = m.groups()

        p_oids = re.findall(r'\b(\d+)\b', meta)
        # Map original parent IDs to new line numbers
        parents = [str(memo[p][0]) for p in p_oids if p in memo and 'input' not in meta]

        # Dedupe: Skip if formula is identical to single parent
        if len(parents) == 1 and memo[p_oids[0]][1] == form:
            memo[oid] = memo[p_oids[0]]; continue

        if 'input' in meta:
            input_num = re.search(r'\d+', meta)
            ctx = "assumption" if "hyp" in meta else f"input {input_num.group()}"
        else:
            ctx = f"{meta.split()[0]} {', '.join(parents)}"

        memo[oid] = (len(lines) + 1, form)
        lines.append(f"{len(lines)}. [{ctx}] {form}")

    return "\n".join(lines)

    
class LogicNLI(Task):

    def __init__(self, config=LogicConfig()):
        super().__init__(config=config)
        self.names = NAMES[:self.config.n_names]
        self.adjectives = ADJECTIVES[:self.config.n_adjectives]
        self.G = fc.partial(FOL_grammar, names=self.names, adjs=self.adjectives)
        self.hyps, self.hyps_weights=make_hyps(self.G)
        self.balancing_key_ratio=1/3

    def generate(self):
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
                print("WARNING","\n".join(proofs))
                continue
            meta.prem, meta.hyp = x.dict(), hyp.dict()
            return Problem(meta, label)

    def prompt(self, meta):
        prem, hyp = meta.prem.eng, meta.hyp.eng
        P = (
            f"Premise:\n{prem}\n"
            f"Hypothesis:\n{hyp}\n\n"
            "If the Premise entails the Hypothesis, the label is 'entailment'.\n"
            "If the Premise contradicts the Hypothesis, the label is 'contradiction'.\n"
            "If neither, the label is 'neutral'.\n"
            "Answer with exactly one word, neutral|contradiction|entailment"
        )

        P=verbalize_predicates(P, seed=meta.verbalize_seed)
        return P

    def balancing_key(self, problem):
        return problem.answer

class EvidenceRetrieval(Task):
    def __init__(self, config=LogicConfig()):
        super().__init__(config=config)
        self.nli = LogicNLI(config=config)

    @staticmethod
    def compute_necessity(x):
        proof_lines = x.metadata.proof.input.splitlines()
        changes = dict()    
        for prefix in [f"fof({i}" for i in x.metadata.proof.indices]:
            ablation = [p for p in proof_lines if not p.startswith(prefix)] 
            y=run("\n".join(ablation))
            changes[prefix]=y.status
        return set(changes.values())=={"Satisfiable"}

    def generate(self):
        while True:
            self.nli.config = self.config
            x = self.nli.generate()
            x.metadata.label=x.answer
            if x.answer != 'neutral' and self.compute_necessity(x):
                break

        answer = [i for i in x.metadata.proof.indices if i != 'hyp']
        answer = ', '.join([f'{i}' for i in answer])
        answer = f'[{answer}]'
        return Problem(x.metadata, answer)

    def prompt(self, meta):
        prem_lines = [f"[{i}] {line}" for i, line in enumerate(meta.prem.eng.splitlines())]
        prem = '\n'.join(prem_lines)
        hyp = meta.hyp.eng
        verb = {'entailment':'entail','contradiction':'contradict'}.get(meta.label)
        P = (
            f"Premise:\n{prem}\n"
            f"Hypothesis:\n{hyp}\n\n"
            f"Which statements in the premise {verb} the hypothesis?\n"
            f"Only answer the list of supporting statements, e.g. [0, 6, 7]."
        )
        P=verbalize_predicates(P, seed=meta.verbalize_seed)
        return P
    
    def score_answer(self, answer, entry):
        reference = entry['answer']
        prepr = lambda x: set(s.strip() for s in x.strip('[].').split(',') if s.strip())
        reference, answer = prepr(reference), prepr(answer)
        if not answer:
            return 0.0
        return len(answer & reference) / len(answer | reference)

    def balancing_key(self, problem):
        return None
        return len(problem.metadata.proof.indices)

