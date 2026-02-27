import numpy as np
import random
import pandas as pd
import networkx as nx
import numbers
from itertools import product
from math import log, nan, floor, ceil
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

# --- The Underlying Object Class (Adapted for pgmpy) --- 🔱
class ReasoningGraph:
    def __init__(self, bn: DiscreteBayesianNetwork = None):
        if bn:
            self.bn = bn.copy()
        else:
            self.bn = DiscreteBayesianNetwork()
        self.reset_inference()

    def generate_new_graph(self, n=4, max_domain_size = 3,**kwargs):
        method = kwargs.pop('method', 'erdos')
        seed = kwargs.pop('seed', None)
        conditionning_seed = kwargs.pop('conditionning_seed', None)
        edge_prob = kwargs.pop('edge_prob', 0.5)
        self.bn = DiscreteBayesianNetwork.get_random(
        n_nodes = n,
        edge_prob = edge_prob,
        n_states = [ k+1 for k in range(1,max_domain_size)],
        method = method,
        seed = seed,
        conditionning_seed = conditionning_seed,
        **kwargs)

        self.ie = CausalVE(self.bn) #Causal Variable Elimination Home Made

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
        return self.ie.query( variables = [self.target], evidence = self.evidence_values, do = self.do_values )

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

    def to_NL(self, n_round: int = 4, verbose = False) -> str:     
        return self.bn.to_nl(n_round, verbose)

    def convert_complex_nodes_to_ci(
        self, 
        cpt_relative_threshold: float, 
        seed: int = None,
        binari_ci_modes: List[str] = ['or', 'and'],
        multi_ci_modes: List[str] = ['max', 'min'],
        n_round: Optional[int] = None,
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
                if n_round != None:
                    new_cpd.round(n_round)
                if new_cpd:
                    self.bn.remove_cpds(old_cpd)
                    self.bn.add_cpds(new_cpd)
                    converted_nodes.append(node)


        self.ie = CausalVE(self.bn)
            
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
    Noisy_mode : bool
        Whether to use the sparser Noisy interaction within the network.
    cpt_relative_threshold : float
        If Noisy_mode is True, set the conversion relative threshold in gain of parameter size (classical/Noisy), to converte a classical CDP interaction into a random Noisy one.
    cot_scientific_notation : bool
        Whether to use scientific notation for chain of thought.
    generate_trivial : bool
        Whether to accept problem where no computationn only retriavial skills are necessary (mainly usefull for law level problems).
    is_verbose : bool
        Whether to use a more humanlike description of the system, or a less verbose one that describe the Bayesian Network by listing all the conditional probabilities.
    """
    n_nodes: int = 3
    max_domain_size: int = 2
    edge_prob: float = 0.5
    graph_generation_mode: str = "erdos"
    n_round: int = 1
    Noisy_mode = True
    cpt_relative_threshold: float = 0
    cot_scientific_notation: bool = False
    graph_seed = None
    conditionning_seed = None
    seed = None
    is_verbose: bool = False
    concise_cot: bool  = True

    def set_level(self, i: int):
        # 1. Call the parent to handle the standard progression logic
        super().set_level(i)
        
        # 2. Handle specific levels
        if i == 0:
            self.generate_trivial = False
        else:
            self.generate_trivial = True
        return self

    def update(self, c):
            self.n_round += .5 * c
            self.n_nodes += .5 * c
            self.max_domain_size += .5 * c
            self.cpt_relative_threshold += .5 * c 


    def set_seed(self, graph_seed = None, conditionning_seed = None):
        """
        Sets the random seeds for reproducibility of the graph structure and the specific query.

        Parameters
        ----------
        Graph_seed : int, optional
            The seed used to control the generation of the Bayesian Network structure 
            (topology, edges) and the parameters (CPDs). If provided, it ensures 
            the "laws of the world" are consistent across runs.
        conditionning_seed : int, optional
            The seed used to control the selection of the target variable, 
            evidence variables, and intervention values (the "scenario"). 
            Allows generating different questions/conditions on the exact same graph.
            Morever, the conditionning accross rungs (association/intervention) are twined.
        """
        self.graph_seed = graph_seed
        self.seed = graph_seed

        self.conditionning_seed = conditionning_seed


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
        n_round = self.config.n_round #stochastic rounding value, that we will use for all subfunction to be coherent within the example    
        self._generate_network(n=self.config.n_nodes,
            method=self.config.graph_generation_mode,
            edge_prob=self.config.edge_prob,
            max_domain_size = self.config.max_domain_size,
            n_round = n_round,
           )

        self._generate_specific_problem(n_round)
        
        answer, specific_metadata = self._calculate_answer_and_metadata(n_round)
        cot = self.reason_graph.ie.generate_natural_language_proof(scientific_notation=self.config.cot_scientific_notation, precision=n_round, concise=self.config.concise_cot)
        while nan in set(eval(answer).values()): #Create another scenario if this one is probabilistically impossible.
            if self.config.graph_seed != None:
                self.config.graph_seed += 1
            self._generate_specific_problem(n_round)
            answer, specific_metadata = self._calculate_answer_and_metadata(n_round)
        
        scenario = self._construct_scenario()
        target_vals = self.reason_graph.bn.states[self.reason_graph.target]

        writer = CanonicalBIFWriter(self.reason_graph.bn)
        bif_data = writer.write_string()

        if 'nan' in cot: #case where the problem is tricky for concise Cot solving
            cot = None
        
        metadata = {
            "target_var_values": target_vals,
            "bif_description":bif_data,
            "scenario": scenario,
            "target": self.reason_graph.target,
            "variables": list(self.reason_graph.bn.nodes()),
            "n_round": n_round,
            "cot": cot
        }
        metadata.update(specific_metadata)
        
        return Problem(metadata=metadata, answer=answer)

    def prompt(self, metadata):
        bif_data = metadata["bif_description"]
        model = ReasoningGraph(CanonicalBIFReader(string=bif_data).get_model())
        n_round = metadata['n_round']
        system_description = model.to_NL(n_round, self.config.is_verbose) 

        target = metadata["target"]
        values = metadata["target_var_values"]

        return (
            f"System:\n{system_description}\n"
            f"Observed conditions:\n{metadata['scenario']}\n"
            f"Task: Compute probability distribution for {target} (possible values: {values}).\n\n"
            f"Output: Python dict mapping each value to its probability, rounded to {n_round} decimals.\n"
            f"Example: {{0: {round(0.123456789,n_round)}, 1: {round(0.876543211,n_round)}}}"
        )

    def get_cot(self, expr):
        return expr.cot

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

    def _generate_network(self,**kwargs):
        self.reason_graph.generate_new_graph(seed= self.config.seed,
         graph_seed = self.config.graph_seed,
         **kwargs)

    def _generate_specific_problem(self,n_round):   
        if self.config.Noisy_mode:     
            self.reason_graph.convert_complex_nodes_to_ci(
                cpt_relative_threshold=self.config.cpt_relative_threshold,
                seed=self.config.graph_seed,
                n_round=n_round,
                )

        if self.config.conditionning_seed:
            self.reason_graph.generate_bonded_rung1(self.config.conditionning_seed)
        else:
            self.reason_graph.generate_rung1()

    def _calculate_answer_and_metadata(self, n_round):
        pred_factor = self.reason_graph.predict()
        answer = str(factor_to_dict(pred_factor, n_round))
        return answer, {}

    def _construct_scenario(self):
        return self.reason_graph.evidences_to_NL()


class BayesianIntervention(Rung, Task):
    def __init__(self, config=Rung12Config()):
        super().__init__(config=config)

    def _generate_network(self,**kwargs):
        self.reason_graph.generate_new_graph(seed = self.config.seed,
         graph_seed = self.config.graph_seed,
          **kwargs)
       
    def _generate_specific_problem(self,n_round): 
        if self.config.Noisy_mode:       
            self.reason_graph.convert_complex_nodes_to_ci(
                cpt_relative_threshold=self.config.cpt_relative_threshold,
                seed=self.config.graph_seed,
                n_round=n_round,
                )

        if self.config.conditionning_seed:
            self.reason_graph.generate_bonded_rung2(self.config.conditionning_seed)
        else:
            self.reason_graph.generate_rung2()

    def _calculate_answer_and_metadata(self, n_round):
        pred_factor = self.reason_graph.predict()
        answer = str(factor_to_dict(pred_factor, n_round))
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

def js_reward(dg, dt, power=128):
    """reward of guessing dg where the true distribution is dt"""
    js = js_divergence(dg, dt)
    return (1 - js / log(2)) ** power