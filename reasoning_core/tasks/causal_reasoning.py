import numpy as np
import random
import pandas as pd
import networkx as nx
import numbers
from itertools import product
from math import log, nan
import ast
from abc import ABC, abstractmethod

from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# --- pgmpy Core Imports ---
from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, CausalInference, BeliefPropagation
from pgmpy.readwrite import BIFWriter
import copy
import types
import logging

from ._causal_utils import *


# --- Monkey patching for pgmpy --- 🐒


def size_full(self):
    return np.prod(self.cardinality)

def size_CI_model(self):
    return np.sum(self.cardinality)



DiscreteBayesianNetwork.get_random = get_random_DBN
DAG.get_random = get_random_DAG
DiscreteBayesianNetwork.to_nl = to_nl_DBN
TabularCPD.to_nl = to_nl_CPD
TabularCPD.size_full = size_full
TabularCPD.size_CI_model = size_CI_model
BinaryInfluenceModel.to_nl = to_nl_BIM
MultilevelInfluenceModel.to_nl = to_nl_MIM
CausalInference.query = query_surgery

# --- The Underlying Object Class (Adapted for pgmpy) --- 🔱
class ReasoningGraph:
    def __init__(self, bn: DiscreteBayesianNetwork = None):
        if bn:
            self.bn = bn.copy()
        else:
            self.bn = DiscreteBayesianNetwork()
        self.reset_inference()

    def generate_new_graph(self, n=4, edge_prob= 0.5, max_domain_size = 3, method="erdos",**kwargs):
        self.bn = DiscreteBayesianNetwork.get_random(
        n_nodes = n,
        edge_prob = edge_prob,
        n_states = [ k+1 for k in range(1,max_domain_size)],
        method = method,
        **kwargs)

        self.ie = CausalInference(self.bn)

    def reset_inference(self):
        self.target = None
        self.do_var = None
        self.evidence_values = {}
        self.do_values = {}

    def generate_rung1(self, seed=None):
        """Sets up an observational query (Rung 1)."""

        self.reset_inference()

        # Local RNG → isolated seed, no global contamination
        rng = random.Random(seed)

        variables = list(self.bn.nodes())
        self.target = rng.choice(variables)

        variables.remove(self.target)
        n_evidence = rng.randint(0, len(variables))
        evidence_variables = rng.sample(variables, n_evidence)

        for state in evidence_variables:
            possible_values = self.bn.states[state]
            self.evidence_values[state] = rng.choice(possible_values)

    def generate_rung2(self, seed=None):
        """Sets up an interventional query (Rung 2)."""

        self.reset_inference()

        rng = random.Random(seed)


        variables = list(self.bn.nodes())
        self.target = rng.choice(variables)

        variables.remove(self.target)

        do_var = rng.choice(variables)
        possible_values = self.bn.states[do_var]
        self.do_values = {do_var: rng.choice(possible_values)}
        self.do_var = do_var

        variables.remove(do_var)

        n_evidence = rng.randint(0, len(variables))
        evidence_variables = rng.sample(variables, n_evidence)

        for state in evidence_variables:
            possible_values = self.bn.states[state]
            self.evidence_values[state] = rng.choice(possible_values)

#### Bounded Generation ####

    #non_root_variables = [node for node in self.bn.nodes() if self.bn.get_parents(node)]

    def generate_bounded_rung1_and_rung2(self, seed=None):
        """Generate matched Rung1 and Rung2 queries from the same seed."""
        rng = random.Random(seed)
        variables = list(self.bn.nodes())

        target = rng.choice(variables)

        remaining_vars = variables.copy()
        remaining_vars.remove(target)

        n_evidence = rng.randint(1, len(remaining_vars)) # up to len(remaining_var) - 1
        evidence_vars = rng.sample(remaining_vars, n_evidence)

        evidence_values = {}
        for v in evidence_vars:
            evidence_values[v] = rng.choice(self.bn.states[v])

        do_var = rng.choice(evidence_vars)
        do_value = evidence_values[do_var]

        # Rung2 evidence is Rung1 evidence minus the promoted do-var
        evidence_vars_r2 = [v for v in evidence_vars if v != do_var]
        evidence_values_r2 = {v: evidence_values[v] for v in evidence_vars_r2}

        self._r1_target = target
        self._r1_evidence_values = evidence_values

        self._r2_target = target
        self._r2_do_var = do_var
        self._r2_do_values = {do_var: do_value}
        self._r2_evidence_values = evidence_values_r2

    def generate_bonded_rung1(self, seed=None):
        """Builds Rung1 using precomputed aligned data."""
        self.generate_bounded_rung1_and_rung2(seed)

        self.reset_inference()
        self.target = self._r1_target
        self.evidence_values = self._r1_evidence_values.copy()


    def generate_bonded_rung2(self, seed=None):
        """Builds Rung2 using precomputed aligned data."""
        self.generate_bounded_rung1_and_rung2(seed)

        self.reset_inference()
        self.target = self._r2_target
        self.do_values = self._r2_do_values.copy()
        self.do_var = self._r2_do_var
        self.evidence_values = self._r2_evidence_values.copy()

#### End bounded generation ####

    def predict(self) -> DiscreteFactor:
        """Make observational predictions."""
        if self.ie is None:
            raise Exception("Inference engine not initialized. Generate a graph first.")
        return self.ie.query([self.target], evidence=self.evidence_values, do = self.do_values)

    def do_to_NL(self):
        """Convert interventional evidence to NL."""
        ret = ""
        if self.do_values:
            ret += "Doing/Imposing that "
            ret += ", and ".join(f"the state {state} is equal to {repr(val)}" 
                                 for state, val in self.do_values.items())
        return ret

    def evidences_to_NL(self):
        """Convert observational evidence to NL."""
        ret = ""
        if self.evidence_values:
            ret += "Observing/Knowing that "
            ret += ", and ".join(f"the state {state} is equal to {repr(val)}" 
                                 for state, val in self.evidence_values.items())
        else:
            ret = "Without further Observation/Knowledge of other variable."
        return ret

    def target_to_NL(self):
        return f"""Provide the probability over the state named {self.target} """

    def to_NL(self, n_round: int = 4, minimalist : bool = False, random_minimalist : bool = True) -> str:     
        return self.bn.to_nl(n_round, minimalist, random_minimalist)

    def convert_complex_nodes_to_ci(
        self, 
        cpt_relative_threshold: float, 
        seed: int = None,
        binari_ci_modes: List[str] = ['or', 'and'],
        multi_ci_modes: List[str] = ['max', 'min'],
    ) -> list:
        """
        Post-processing step to convert large TabularCPDs to CI models.
        """
        if self.bn is None or not self.bn.nodes():
            return []
            
        rng = np.random.default_rng(seed)
        converted_nodes = []
        
        for node in self.bn.nodes():
            old_cpd = self.bn.get_cpds(node)
            
            if not isinstance(old_cpd, TabularCPD):
                continue

            cpt_relative_diff = (old_cpd.size_full() - old_cpd.size_CI_model())/old_cpd.size_CI_model()
            
            if cpt_relative_diff > cpt_relative_threshold:
                parents = list(self.bn.predecessors(node))
                if not parents:
                    continue

                all_vars = [node] + parents
                cardinalities = {v: self.bn.get_cardinality(v) for v in all_vars}
                child_card = cardinalities[node]
                parent_cards = [cardinalities[p] for p in parents]
                
                new_cpd = None

                if child_card == 2 and all(c == 2 for c in parent_cards):
                    chosen_mode = rng.choice(binari_ci_modes)
                else:
                    chosen_mode = rng.choice(multi_ci_modes)
                new_cpd = get_random_CI(
                        variable = node,
                        evidence = parents,
                        cardinality = cardinalities,
                        mode = chosen_mode,
                        seed = seed)
                if new_cpd:
                    self.bn.remove_cpds(old_cpd)
                    self.bn.add_cpds(new_cpd)
                    converted_nodes.append(node)


        self.ie = CausalInference(self.bn)
            
        return converted_nodes


# --- Causal generator class (Adapted for pgmpy) --- 🏡

@dataclass
class Rung12Config(Config):
    """
    Configuration for Rung 1 and Rung 2 tasks.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    max_domain_size : int
        Maximum domain size for the variables.
    edge_prob : float
        Probability of an edge between two nodes.
    graph_generation_mode : str
        Method for generating the graph (e.g., "erdos").
    n_round : int
        Number of decimal places to round probabilities to.
    """
    n_nodes: int = 3
    max_domain_size: int = 2
    edge_prob: float = 0.5
    graph_generation_mode: str = "erdos"
    n_round: int = 2
    Graph_seed = None
    Conditionning_seed = None
    seed = None
    Noisy_mode = True
    cpt_relative_threshold: float = 0

    def update(self, c):
        self.n_nodes+= c
        self.max_domain_size+= 0.5 * c
        self.cpt_relative_threshold += 0.5 * c 

    def set_seed(self, Graph_seed = None, Conditionning_seed = None):
        self.Graph_seed = Graph_seed
        self.seed = Graph_seed

        self.Conditionning_seed = Conditionning_seed


class Rung(ABC):
    """An abstract base class for Rung tasks of any degree."""
    def __init__(self, config=Rung12Config(), bn: DiscreteBayesianNetwork = None):
        super().__init__()
        self.config = config
        self.reason_graph = ReasoningGraph(bn=bn)

    @abstractmethod
    def _generate_specific_problem(self):
        pass

    @abstractmethod
    def _generate_network(self, **kwargs):
        pass

    @abstractmethod
    def _calculate_answer_and_metadata(self):
        pass

    @abstractmethod
    def _construct_scenario(self):
        pass

    def generate(self):
        
        self._generate_network(n=self.config.n_nodes,
            method=self.config.graph_generation_mode,
            edge_prob=self.config.edge_prob,
            max_domain_size = self.config.max_domain_size,
           )

        self._generate_specific_problem()
        
        answer, specific_metadata = self._calculate_answer_and_metadata()

        #while nan in set(eval(answer).values()): #Create another scenario if this one is probabilistically impossible.
        #    self._generate_specific_problem()
        #    answer, specific_metadata = self._calculate_answer_and_metadata()
        
        system_description = self.reason_graph.to_NL(self.config.n_round)
        scenario = self._construct_scenario()
        target_vals = self.reason_graph.bn.states[self.reason_graph.target]

        writer = CanonicalBIFWriter(self.reason_graph.bn)
        bif_data = writer.write_string()
        
        metadata = {
            "target_var_values": target_vals,
            "bif_description":bif_data,
            #"system_description": system_description,
            "scenario": scenario,
            "target": self.reason_graph.target,
            "variables": list(self.reason_graph.bn.nodes())
        }
        metadata.update(specific_metadata)
        
        return Problem(metadata=metadata, answer=answer)

    def prompt(self, metadata):
        bif_data = metadata["bif_description"]
        model = ReasoningGraph( CanonicalBIFReader( string = bif_data ).get_model() )
        system_description = model.to_NL(self.config.n_round)

        #system_description = metadata["system_description"]
        scenario = metadata["scenario"]
        target_variable = metadata["target"]
        target_var_values = metadata["target_var_values"]

        output_format_instructions = (
            "You must return the probability distribution over all values of the target variable "
            "in the format of a Python dictionary. The output should map each value to its estimated probability.\n"
            "You will be evaluated based on how close your estimated probability distribution is to the true one.\n\n"
            "For example, if the target variable is V01 (which can take values 0 or 1) "
            "and you estimate that P(V01 = 0) = 0.4 and P(V01 = 1) = 0.6, "
            "your answer must be: {0: 0.4, 1: 0.6} "
        )
        task_description = (
            f"Calculate the probability distribution for the target variable {target_variable}, "
            f"which can take the following values: {target_var_values}."
        )
        return (
            f"### System Description\n"
            f"""Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:\n"""
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
            dict_pred = _to_dict(answer)
        except:
            return 0
        return js_reward(dict_truth, dict_pred)


class BayesianAssociation(Rung, Task):
    def __init__(self, config=Rung12Config()):
        super().__init__(config=config)

    def _generate_network(self, method="erdos",**kwargs):
        self.reason_graph.generate_new_graph(method=method, seed=self.config.Graph_seed, **kwargs)

    def _generate_specific_problem(self):   
        if self.config.Noisy_mode:     
            self.reason_graph.convert_complex_nodes_to_ci(
                cpt_relative_threshold=self.config.cpt_relative_threshold,
                seed=self.config.Graph_seed,
                )

        if self.config.Conditionning_seed:
            self.reason_graph.generate_bonded_rung1(self.config.Conditionning_seed)
        else:
            self.reason_graph.generate_rung1()

    def _calculate_answer_and_metadata(self):
        pred_factor = self.reason_graph.predict()
        answer = str(factor_to_dict(pred_factor, self.config.n_round))
        return answer, {}

    def _construct_scenario(self):
        return self.reason_graph.evidences_to_NL()


class BayesianIntervention(Rung, Task):
    def __init__(self, config=Rung12Config()):
        super().__init__(config=config)

    def _generate_network(self, method="erdos",**kwargs):
        self.reason_graph.generate_new_graph(method=method, seed=self.config.Graph_seed, **kwargs)
        
    def _generate_specific_problem(self): 
        if self.config.Noisy_mode:       
            self.reason_graph.convert_complex_nodes_to_ci(
                cpt_relative_threshold=self.config.cpt_relative_threshold,
                seed=self.config.Graph_seed,
                )

        if self.config.Conditionning_seed:
            self.reason_graph.generate_bonded_rung2(self.config.Conditionning_seed)
        else:
            self.reason_graph.generate_rung2()

    def _calculate_answer_and_metadata(self):
        pred_factor = self.reason_graph.predict()
        answer = str(factor_to_dict(pred_factor, self.config.n_round))
        return answer, {}

    def _construct_scenario(self):
        doing = self.reason_graph.do_to_NL()
        seeing = self.reason_graph.evidences_to_NL()
        
        parts = [part for part in [doing, seeing] if part and 
                 part != "Without further Observation/Knowledge of other variable."]
        
        if not parts:
            return "Without any intervention or observation."
        return ". ".join(parts)


# --- functions for score computation (Adapted) --- 💯

def factor_to_dict(factor: TabularCPD, n_round: int = 2) -> dict:
    """Converts a 1D pgmpy posterior factor into a result dict."""
    if len(factor.variables) != 1:
        raise ValueError("Factor must be a 1D posterior distribution.")
    
    var = factor.variables[0]
    states = factor.state_names[var]
    values_rounded = [round(val, n_round) for val in  factor.values]
    return _to_dict({state: float(val) for state, val in zip(states, values_rounded)})


def _to_dict(x):
    """Converts a string representation or dict into a standard dict."""
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError, Exception):
            raise TypeError(f"Could not parse string: {x}")

    if not isinstance(x, dict):
        raise TypeError(f"Expected a dict (or its string repr), got {type(x)}")

    out = {}
    for k, v in x.items():
        try:
            k2 = int(k) if isinstance(k, str) and k.isdigit() else k
        except (ValueError, TypeError):
            k2 = k
        out[k2] = float(v)
    
    total = sum(out.values())
    if total > 0 and not np.isclose(total, 1.0):
        for k in out:
            out[k] = out[k] / total
            
    return out


def js_divergence(d1, d2):
    """
    Compute the Jensen-Shannon divergence between two discrete probability distributions.
    """
    keys = set(d1.keys()).union(set(d2.keys()))
    if not keys:
        return 0.0
        
    p = [d1.get(k, 0.0) for k in keys]
    q = [d2.get(k, 0.0) for k in keys]
    
    p_sum = sum(p)
    q_sum = sum(q)
    if p_sum > 0: p = [v / p_sum for v in p]
    if q_sum > 0: q = [v / q_sum for v in q]
    
    m = [(p[i] + q[i]) / 2 for i in range(len(keys))]

    def kl_divergence(a, b):
        return sum(a_i * log(a_i / b_i, 2) for a_i, b_i in zip(a, b) if a_i > 0 and b_i > 0)

    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return js

def js_reward(dg, dt, power=512):
    """reward of guessing dg where the true distribution is dt"""
    js = js_divergence(dg, dt)
    return (1 - js / log(2)) ** power