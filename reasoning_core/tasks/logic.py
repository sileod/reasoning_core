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

from ._logic_utils import cat_premises, satify_premise


eng, tptp = "eng","tptp"

ADJECTIVES = ['rich', 'quiet', 'old', 'tall', 'kind', 'brave', 'wise',
              'happy', 'strong', 'curious', 'patient', 'funny', 'generous', 'humble']

NAMES = ['mary', 'paul', 'fred', 'alice', 'john', 'susan', 'lucy']

G = FOL_grammar

def make_hyps(N=1000):
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


def verbalize_predicates(x):
    preds = random.sample(fol_nli_verbalization.predicates, len(preds_pattern))
    npreds = [fol_nli_verbalization.negate_predicate(p) for p in preds]
    sub=dict()
    sub|=dict(zip(npreds_pattern,npreds))
    sub|=dict(zip(preds_pattern,preds))

    for k,v in sub.items():
        x=x.replace(k,v)
    return x.replace('_',' ')

def valid(x):
    for p in "", "~":
        status= run(f"fof(f,axiom,{p}({x@tptp})).").status
        assert  status in ["Satisfiable", "Unsatisfiable", "Refutation not found", "Time limit"]
        if status!="Satisfiable":
            return False
    return True


@dataclass
class LogicConfig(Config):
    n_formulas: int = 8 
    generation_algorithm: str = "sequential"
    def update(self, c):
        self.n_formulas *= (1 + c)



class LogicNLI(Task):

    def __init__(self, config=LogicConfig()):
        super().__init__(config=config)
        self.hyps, self.hyps_weights=make_hyps()

    def generate(self):
        meta = edict()
        # generate premise
        x = generate_N_premises(self.config.n_formulas, G, mode=self.config.generation_algorithm)
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
        meta.proof = proof = ([x for x in proofs if x.status=="Unsatisfiable"]+[None])[0]
        labels = tuple([x.status for x in proofs])

        label = {
            ('Satisfiable', 'Unsatisfiable'): 'entailment',
            ('Satisfiable', 'Satisfiable'): 'neutral',
            ('Unsatisfiable', 'Satisfiable'): 'contradiction'
        }.get(labels,'other')

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

        P=verbalize_predicates(P)
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
        return verbalize_predicates(P)

    def score_answer(self, answer, entry):
        reference = entry['answer']
        prepr = lambda x: set(s.strip() for s in x.strip('[].').split(',') if s.strip())
        reference, answer = prepr(reference), prepr(answer)
        if not answer:
            return 0.0
        return len(answer & reference) / len(answer | reference)

    def balancing_key(self, problem):
        return len(problem.metadata.proof.indices)

