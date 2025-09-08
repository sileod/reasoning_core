import pyagrum
gum = pyagrum
import pyagrum.causal as csl
import numpy as np
import random
import math
import ast

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
        ret = ""
        if self.do_values:
            ret += "Doing/Imposing that "
            ret += ", and ".join(f"the state {state} is equal to {val}" for state, val in self.do_values.items())
        return ret


    def evidences_to_NL(self):
        ret = ""
        if self.evidence:
            ret += "Observing/Knowing that "
            ret += ", and ".join(f"the state {state} is equal to {val}" for state, val in self.evidence.items())
        return ret

    def target_to_NL(self):
        return f"""Provide the probability over the state named {self.target} """


    def display_CPTs(self):
        """Show the CPTs of the model, but only works in notebooks """
        bn = self.bn
        tables = [bn.cpt(node) for node in bn.names()]
        gnb.sideBySide(*tables)

    def show_CPTs(self):
        bn = self.bn
        for node in bn.names():
            print(f"--- CPT for {node} ---")
            print(bn.cpt(node))  # Displays the CPT in text form
            print("\n")

    def to_NL( self, n_round : int = 4, minimalist = False, random_minimalist = True) -> str :
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
        
        return " \n ".join(descriptions)


# --- Causal generator class  --- 🏡

@dataclass
class Rung12Config(Config):
    n: int = 4
    domain_size: int = 2

    def update(self, c):
        self.n+=c
        self.domain_size+=c

class Rung2(Task):
    def __init__(self, config=Rung12Config(), bn : gum.BayesNet = None):
        super().__init__(config=config)
        self.reason_graph = ReasoningGraph(bn = bn)

    def generate_new_problem(self, n=4, domain_size=2):
        self.reason_graph.generate_new_graph(n = n,domain_size = domain_size)
        self.reason_graph.generate_rung2()

    def prompt(self, metadata):
        target_var_values = metadata["target_var_values"]
        premise = metadata["premise"]
        hypothesis = ". ".join([metadata["doing"], metadata["seeing"]])
        target = metadata["target"]

        instruction = (
            "We will provide you a target variable, and you must return its distribution over all its values, "
            "in the format of a Python dictionary. The output should map each value to its estimated probability\n"
            "Be concise, and strictly follow the required output format. You will be evaluated based on how close your "
            "estimated probability distributions are to the true ones.\n\n"
        )

        format_exemple = (
            "As an example, if you are asked to estimate the distributions for variables X01 (which can take values 0 or 1)"
            ", and you estimate that:\n"
            "- P(X01 = 0) = 0.4, P(X01 = 1) = 0.6\n"
            "Then your answer must be: <xml>{0: 0.4, 1: 0.6}</xml>. "
            "However, if a separate instruction asks for XML field replacement tags, use that format instead (e.g., <answer>{0: 0.4, 1: 0.6}</answer>).\n\n"
        )

        explicit_question = "You are asked to estimate the following variable: "
        explicit_question += f" the variable {target}, which can take the values {target_var_values}."

        prompt = (
            f"Premise:\n{premise}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Instructions:\n{instruction}\n\n"
            f"Example and format:\n{format_exemple}\n\n"
            f"{explicit_question}"
        )

        return prompt

    def generate(self):
        self.generate_new_problem( n=self.config.n, domain_size=self.config.domain_size)
        rg = self.reason_graph
        formula, pred, explanation = csl.causalImpact(rg.causalbn, rg.target, rg.do_var, knowing = rg.observed_vars, values = rg.do_values | rg.evidence)

        answer = str(tensor_to_dict_1d( pred ))

        premise = rg.to_NL(n_round=2)
    
        seeing = rg.evidences_to_NL()
        doing = rg.do_to_NL()

        var = rg.target
        target_vals = list(rg.bn.variable(var).labels())

        metadata = {}
        metadata["target_var_values"] = target_vals
        metadata["premise"] = premise
        metadata["seeing"] = seeing
        metadata["doing"] = doing
        metadata["target"] = rg.target
        metadata["variables"] = list(rg.bn.names())
        metadata["formula"] = formula.toLatex()
        metadata["explanation"] = explanation
        return Problem( metadata = metadata, answer = answer)

    def score_answer(self, answer, entry):
        dict_truth = _to_dict(entry.answer)
        try:
            dict_pred  = _to_dict(answer)
        except:
            return 0
        return js_reward(dict_truth, dict_pred)
        

class Rung1(Task):
    def __init__(self, config=Rung12Config(), bn : gum.BayesNet = None):
        super().__init__(config=config)
        self.reason_graph = ReasoningGraph(bn = bn)

    def generate_new_problem(self, n=4, domain_size=2):
        self.reason_graph.generate_new_graph(n = n,domain_size = domain_size)
        self.reason_graph.generate_rung1()

    def prompt(self, metadata):
        target_var_values = metadata["target_var_values"]
        premise = metadata["premise"]
        hypothesis = metadata["hypothesis"]
        target = metadata["target"]

        instruction = (
            "We will provide you a target variable, and you must return its distribution over all its values, "
            "in the format of a Python dictionary. The output should map each value to its estimated probability\n"
            "Be concise, and strictly follow the required output format. You will be evaluated based on how close your "
            "estimated probability distributions are to the true ones.\n\n"
        )

        format_exemple = (
            "As an example, if you are asked to estimate the distributions for variables X01 (which can take values 0 or 1)"
            ", and you estimate that:\n"
            "- P(X01 = 0) = 0.4, P(X01 = 1) = 0.6\n"
            "Then your answer must be: <xml>{0: 0.4, 1: 0.6}</xml>. "
            "However, if a separate instruction asks for XML field replacement tags, use that format instead (e.g., <answer>{0: 0.4, 1: 0.6}</answer>).\n\n"
        )

        explicit_question = "You are asked to estimate the following variable: "
        explicit_question += f" the variable {target}, which can take the values {target_var_values}."

        prompt = (
            f"Premise:\n{premise}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Instructions:\n{instruction}\n\n"
            f"Example and format:\n{format_exemple}\n\n"
            f"{explicit_question}"
        )

        return prompt

    def generate(self):
        self.generate_new_problem( n=self.config.n, domain_size=self.config.domain_size)

        pred = self.reason_graph.predict()
        answer = str(tensor_to_dict_1d( pred ))

        premise = self.reason_graph.to_NL(n_round=2)
    
        hypothesis = self.reason_graph.evidences_to_NL()

        var_values = {}
        var = self.reason_graph.target
        target_vals = list(self.reason_graph.bn.variable(var).labels())

        metadata = {}
        metadata["target_var_values"] = target_vals
        metadata["premise"] = premise
        metadata["hypothesis"] = hypothesis
        metadata["target"] = list(self.reason_graph.target)
        metadata["variables"] = list(self.reason_graph.bn.names())
        return Problem( metadata = metadata, answer = answer)

    def score_answer(self, answer, entry):
        dict_truth = _to_dict(entry.answer)
        try:
            dict_pred  = _to_dict(answer)
        except:
            return 0
        return js_reward(dict_truth, dict_pred)


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