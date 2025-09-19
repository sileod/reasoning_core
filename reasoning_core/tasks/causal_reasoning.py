import pyagrum as gum
import pyagrum
import pyagrum.causal as csl
import numpy as np
import random
import math
import ast
from abc import ABC, abstractmethod

from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass

# --- The underlying object class (bayesian network) --- 🔱

class ReasoningGraph():
    def __init__(self, bn = None):
        if bn:
            self.bn = gum.BayesNet(bn)
        else:
            self.bn = gum.BayesNet()
        self.reset_inference()


    def generate_new_graph(self, n=4, domain_size=2):
        gen = pyagrum.BNGenerator()
        self.bn = gen.generate(n_nodes= n, n_arcs=int(n * (n-1) / 2), n_modmax = domain_size)
        

    def generate_rung1(self):
        """set the environment of the object in order to later on make rung 1 query"""
        self.reset_inference()
        variables = list(self.bn.names())
        n_variables = len(variables)
        self.target = random.choice(variables)
        variables.remove(self.target) # now variables contain the variables we can condition on
        n_rem = len(variables)
        n_evidence = random.randint(0,n_rem)
        evidence_variables = random.sample(variables, n_evidence)
        for state in evidence_variables:
            possible_values = self.bn.variable(state).labels()
            val = random.choice(possible_values)
            self.evidence[state] = val
        self.set_ie()

    def reset_inference(self):
        """reset the inference engine of the object to the base one (empty one)"""
        self.ie = gum.LazyPropagation(self.bn) #inference engine
        self.target = None
        self.evidence = {}
        

    def generate_rung2(self):
        """set the environment of the object in order to later on make rung 2 query"""
        self.reset_causal()
        variables = list(self.bn.names())
        n_variables = len(variables)
        self.target = random.choice(variables)
        variables.remove(self.target) # now variables contain the variables we can condition on and the causal one
        self.do_var = random.choice(variables)
        val = random.choice(self.bn.variable(self.do_var).labels())
        self.do_values[self.do_var] = val
        variables.remove(self.do_var) # now variables contain the variables we can condition on
        n_rem = len(variables)
        n_evidence = random.randint(0,n_rem)
        evidence_variables = random.sample(variables, n_evidence)
        for state in evidence_variables:
            possible_values = self.bn.variable(state).labels()
            val = random.choice(possible_values)
            self.evidence[state] = val
        self.observed_vars = set(evidence_variables)

    def reset_causal(self):
        """reset the inference engine of the object to the base one (empty one)"""
        self.causalbn = csl.CausalModel(self.bn)
        self.do_var = None
        self.do_values = {}
        self.target = None
        self.observed_vars = set()
        self.evidence = {}

    def set_ie(self):
        """set the inference engine within the environment previously set, in order to make predictions later on"""
        self.ie.setEvidence(self.evidence)
        self.ie.makeInference()

    def predict(self):
        """make predictions over the targets, once the ie is well set"""
        return self.ie.posterior(self.target)

    def do_to_NL(self):
        """convert the ReasoningGraph conditionning to natural language (rung 2 version with the do operator)"""
        ret = ""
        if self.do_values:
            ret += "Doing/Imposing that "
            ret += ", and ".join(f"the state {state} is equal to {val}" for state, val in self.do_values.items())
        return ret


    def evidences_to_NL(self):
        """convert the ReasoningGraph structure and conditionning to natural language (rung 1 version)"""
        ret = ""
        if self.evidence:
            ret += "Observing/Knowing that "
            ret += ", and ".join(f"the state {state} is equal to {val}" for state, val in self.evidence.items())
        else:
            ret = "Without further Observation/Knowledge of other variable."
        return ret

    def target_to_NL(self):
        return f"""Provide the probability over the state named {self.target} """



    def to_NL( self, n_round : int = 4, minimalist = False, random_minimalist = True) -> str :
        """convert the ReasoningGraph structure into natural language"""
        bn = self.bn
        descriptions = []
        for node in bn.names():
            # Use Instantiation to iterate over all configurations of the CPT
            local_cpt = bn.cpt(node)
            I = gum.Instantiation(local_cpt)
            I.setFirst()  # initialize the iterator

            grouped_probs = {}  # Store probabilities grouped by conditions
        
            while not I.end():
                # Build the conditioning description using variable names.
                cond_desc = " and ".join(
                    f"{var.name()} = {I[var.name()]}" 
                    for var in I.variablesSequence() 
                    if var.name() != node
                )
                # Get the probability for the current instantiation.
                prob = round(local_cpt.get(I), n_round)
                # Get the value for the target variable (node) using its name.
                target_val = I[bn.variable(node).name()]
                if cond_desc not in grouped_probs:
                    grouped_probs[cond_desc] = [(target_val, prob)] #initialise the list of conditional values
                else:
                    grouped_probs[cond_desc].append((target_val, prob))
                I.inc()

            for cond, probs in grouped_probs.items():
                if not minimalist:
                    prob_text = " and ".join(f"the probability of {node} = {val} is {prob}" for val, prob in probs)
                else:
                    n = len(probs)
                    drop_i = random.randint(0, n-1)
                    probs.pop(drop_i)
                    prob_text = " and ".join(f"the probability of {node} = {val} is {prob}" for val, prob in probs)
                if cond :
                    descriptions.append(f"If {cond}, then {prob_text}.")
                else:
                    descriptions.append(f"{prob_text}.")
        
        return "  \n".join(descriptions)


# --- Causal generator class  --- 🏡

@dataclass
class Rung12Config(Config):
    n: int = 3
    domain_size: int = 2

    def update(self, c):
        self.n+= c
        self.domain_size+= 0.5 * c

class Rung(ABC):
    """An abstract base class for Runf tasks of any degree."""
    def __init__(self, config=Rung12Config(), bn: gum.BayesNet = None):
        super().__init__(config=config)
        self.reason_graph = ReasoningGraph(bn=bn)

    @abstractmethod
    def _generate_specific_problem(self, n, domain_size):
        """Abstract method for rung-specific problem generation."""
        pass

    @abstractmethod
    def _calculate_answer_and_metadata(self):
        """Abstract method to compute the answer and any extra metadata."""
        pass

    @abstractmethod
    def _construct_scenario(self):
        """Abstract method to construct the natural language scenario/scenario."""
        pass

    def generate(self):
        self._generate_specific_problem(n=self.config.n, domain_size=self.config.domain_size)
        
        answer, specific_metadata = self._calculate_answer_and_metadata()
        
        system_description = self.reason_graph.to_NL(n_round=2)
        scenario = self._construct_scenario()
        target_vals = list(self.reason_graph.bn.variable(self.reason_graph.target).labels())
        
        metadata = {
            "target_var_values": target_vals,
            "system_description": system_description,
            "scenario": scenario,
            "target": self.reason_graph.target,
            "variables": list(self.reason_graph.bn.names())
        }
        metadata.update(specific_metadata)
        
        return Problem(metadata=metadata, answer=answer)

    def prompt(self, metadata):
        system_description = metadata["system_description"]
        scenario = metadata["scenario"]
        target_variable = metadata["target"]
        target_var_values = metadata["target_var_values"]

        # The instruction and format example are now combined.
        output_format_instructions = (
            "You must return the probability distribution over all values of the target variable "
            "in the format of a Python dictionary. The output should map each value to its estimated probability.\n"
            "You will be evaluated based on how close your estimated probability distribution is to the true one.\n\n"
            "For example, if the target variable is X01 (which can take values 0 or 1) "
            "and you estimate that P(X01 = 0) = 0.4 and P(X01 = 1) = 0.6, "
            "your answer must be: {0: 0.4, 1: 0.6} (in between the proper xml tags if asked). "
        )

        # The final question is now the main task description.
        task_description = (
            f"Calculate the probability distribution for the variable '{target_variable}', "
            f"which can take the following values: {target_var_values}."
        )

        return (
            f"### System Description\n"
            f"This section describes the probabilistic relationships between variables in the system:\n"
            f"{system_description}\n\n"
            f"### Scenario\n"
            f"Given the system described above, consider the following specific conditions:\n"
            f"{scenario}\n\n"
            f"### Your Task\n"
            f"{task_description}\n\n"
            f"### Required Output Format\n"
            f"{output_format_instructions}"
        )

    def score_answer(self, answer, entry):
        """Shared scoring function."""
        dict_truth = _to_dict(entry.answer)
        try:
            dict_pred  = _to_dict(answer)
        except:
            return 0
        return js_reward(dict_truth, dict_pred)


class BayesianAssociation(Rung, Task):
    def __init__(self, config=Rung12Config()):
        super().__init__(config=config)
        self.reason_graph = ReasoningGraph()

    def _generate_specific_problem(self, n=4, domain_size=2):
        self.reason_graph.generate_new_graph(n=n, domain_size=domain_size)
        self.reason_graph.generate_rung1()

    def _calculate_answer_and_metadata(self):
        pred = self.reason_graph.predict()
        answer = str(tensor_to_dict_1d(pred))
        return answer, {}  # No specific metadata for Rung1

    def _construct_scenario(self):
        return self.reason_graph.evidences_to_NL()

class BayesianIntervention(Rung, Task):
    def __init__(self, config=Rung12Config()):
        super().__init__(config=config)
        self.reason_graph = ReasoningGraph()
        
    def _generate_specific_problem(self, n=4, domain_size=2):
        self.reason_graph.generate_new_graph(n=n, domain_size=domain_size)
        self.reason_graph.generate_rung2()

    def _calculate_answer_and_metadata(self):
        rg = self.reason_graph
        formula, pred, explanation = csl.causalImpact(
            rg.causalbn, rg.target, rg.do_var, 
            knowing=rg.observed_vars, values=rg.do_values | rg.evidence
        )
        answer = str(tensor_to_dict_1d(pred))
        metadata = {
            "formula": formula.toLatex(),
            "explanation": explanation
        }
        return answer, metadata

    def _construct_scenario(self):
        doing = self.reason_graph.do_to_NL()
        seeing = self.reason_graph.evidences_to_NL()
        # Filter out empty strings before joining
        parts = [part for part in [doing, seeing] if part]
        return ". ".join(parts)



# --- functions for score computation --- 💯
def _to_dict(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    if not isinstance(x, dict):
        raise TypeError(f"Expected a dict (or its string repr), got {type(x)}")

    # cast keys like "0", "1" -> 0, 1
    out = {}
    for k, v in x.items():
        try:
            k2 = int(k) if isinstance(k, str) and k.isdigit() else k
        except (ValueError, TypeError):
            k2 = k
        out[k2] = v
    return out

def tensor_to_dict_1d(tensor):
    return {i: float(tensor[i]) for i in range(tensor.shape[0])}


def js_divergence(d1, d2):
    """
    Compute the Jensen-Shannon divergence between two discrete probability distributions.
    Both d1 and d2 must be dictionaries with the same keys and values summing to 1.
    Returns a value between 0 and log(2).
    """
    keys = set(d1.keys()).union(set(d2.keys()))
    p = [d1.get(k, 0.0) for k in keys]
    q = [d2.get(k, 0.0) for k in keys]
    m = [(p[i] + q[i]) / 2 for i in range(len(keys))]

    def kl_divergence(a, b):
        return sum(a_i * math.log(a_i / b_i, 2) for a_i, b_i in zip(a, b) if a_i > 0)

    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return js

def js_reward(dg, dt, power=512):
    """reward of guessing dg where the true distribution is dt"""
    js = js_divergence(dg, dt)
    return (1 - js / math.log(2)) ** power
