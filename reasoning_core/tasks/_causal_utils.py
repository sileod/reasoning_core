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
from pgmpy.readwrite import BIFWriter, BIFReader
from pgmpy.inference import VariableElimination
from pgmpy.factors import factor_product

import types
import logging

from copy import deepcopy

import json
import re

import copy
import logging
import itertools

# --- Monkey patching for pgmpy --- 🐒
class BinaryInfluenceModel(TabularCPD):
    def __init__(
        self,
        variable,
        evidence,
        activation_magnitude,
        mode="OR",
        leak=None,
        isboolean_style=False,
        state_names=None,
    ):
        self.mode = mode.upper()
        if self.mode not in {"OR", "AND"}:
            raise ValueError("mode must be either 'OR' or 'AND'")

        self.isboolean_style = isboolean_style
        self.activation_magnitude = np.asarray(activation_magnitude, dtype=float)
        self.leak = np.array([leak]) if leak is not None else None
        self.isleaky = leak is not None
        
        # Robustness fixes
        self.evidence = list(evidence)

        if state_names is None:
            state_names = {}
            full_variables = [variable] + self.evidence
            if self.isboolean_style:
                default_states = [False, True]
            else:
                default_states = [0, 1]
            for v in full_variables:
                state_names[v] = default_states
        
        self.full_state_names = state_names

        parent_states = [self.full_state_names[e] for e in self.evidence]
        cols = []
        for combo in product(*parent_states):
            evidence_inst = dict(zip(self.evidence, combo))
            probs = self._evaluate(evidence_inst)
            cols.append(probs)

        cpd_values = np.array(cols).T
        
        super().__init__(
            variable=variable,
            variable_card=2,
            values=cpd_values,
            evidence=self.evidence,
            evidence_card=[2] * len(self.evidence),
            state_names=self.full_state_names,
        )

    def _evaluate(self, evidence_instantiate: dict) -> np.ndarray:
        # Dynamic Active State Detection (Last state = Active)
        active_mask = []
        for e in self.evidence:
            val = evidence_instantiate[e]
            states = self.full_state_names[e]
            active_mask.append(val == states[-1])
        
        active_mask = np.array(active_mask)
        probs = self.activation_magnitude[active_mask]

        if self.mode == "OR":
            p_active = 1 - np.prod(1 - probs)
            if self.isleaky:
                p_active = 1 - (1 - p_active) * (1 - self.leak[0])
        else:  # AND
            if np.any(~active_mask):
                p_active = self.leak[0] if self.isleaky else 0.0
            else:
                p_active = np.prod(probs)
                if self.isleaky:
                    p_active = 1 - (1 - p_active) * (1 - self.leak[0])

        return np.array([1 - p_active, p_active])

    # --- CRITICAL FIX: PRESERVE CLASS TYPE ON COPY ---
    def copy(self):
        new_cpd = BinaryInfluenceModel(
            variable=self.variable,
            evidence=self.evidence,
            activation_magnitude=self.activation_magnitude,
            mode=self.mode,
            leak=self.leak[0] if self.leak is not None else None,
            isboolean_style=self.isboolean_style,
            state_names=self.state_names.copy()
        )
        return new_cpd


class MultilevelInfluenceModel(TabularCPD):
    def __init__(self, variable, evidence, influence_tables, levels, leak=None, mode="MAX", state_names=None):
        self.mode = mode.upper()
        if self.mode not in {"MAX", "MIN"}:
            raise ValueError("mode must be 'MAX' or 'MIN'")
            
        self.levels = levels
        self.influence_tables = influence_tables
        self.leak = np.array(leak) if leak is not None else None
        self.isleaky = leak is not None
        self.evidence = list(evidence)

        if state_names is None:
            state_names = {}
            for parent in self.evidence:
                if parent not in state_names:
                    state_names[parent] = list(influence_tables[parent].keys())
        
        self.full_state_names = state_names

        # ... (Validation and Table Generation Logic from previous turns) ...
        # (Assuming the logic for generating values is same as before)
        
        # Recalculate values for init
        self.cumulative_tables = {
            p: {v: np.cumsum(probs) for v, probs in table.items()}
            for p, table in influence_tables.items()
        }
        self.cumulative_leak = np.cumsum(leak) if leak is not None else np.ones(levels)
        
        parent_states = [self.full_state_names[p] for p in self.evidence]
        cols = []
        for combo in product(*parent_states):
            e = dict(zip(evidence, combo))
            probs = self._evaluate(e)
            cols.append(probs)
        values = np.vstack(cols).T

        super().__init__(
            variable=variable,
            variable_card=self.levels,
            values=values,
            evidence=self.evidence,
            evidence_card=[len(st) for st in parent_states],
            state_names=self.full_state_names,
        )

    # ... (Keep _validate_probs and _evaluate from previous turns) ...
    def _validate_probs(self, arr, name="probabilities"):
        arr = np.asarray(arr)
        if np.any(arr < 0) or np.any(arr > 1):
             raise ValueError(f"{name} must be between 0 and 1.")
        # Relax tolerance slightly for floating point noise
        if not np.isclose(arr.sum(), 1.0, atol=1e-5):
             raise ValueError(f"{name} must sum to 1. Got {arr.sum()}")
        return arr

    def _evaluate(self, evidence_instantiate: dict) -> np.ndarray:
        if self.mode == "MAX":
            cum_prob = np.ones(self.levels)
            for parent, val in evidence_instantiate.items():
                theta = self.cumulative_tables[parent][val]
                cum_prob *= theta
            if self.isleaky:
                cum_prob *= self.cumulative_leak
        elif self.mode == "MIN":
            complement_prod = np.ones(self.levels)
            for parent, val in evidence_instantiate.items():
                theta = self.cumulative_tables[parent][val]
                complement_prod *= (1 - theta)
            if self.isleaky:
                complement_prod *= (1 - self.cumulative_leak)
            cum_prob = 1 - complement_prod

        cum_prob = np.maximum.accumulate(np.clip(cum_prob, 0, 1))
        probs = np.diff(np.concatenate(([0.0], cum_prob)))
        return probs / probs.sum()

    # --- CRITICAL FIX: PRESERVE CLASS TYPE ON COPY ---
    def copy(self):
        new_cpd = MultilevelInfluenceModel(
            variable=self.variable,
            evidence=self.evidence,
            influence_tables=self.influence_tables, # Pass original dict
            levels=self.levels,
            leak=self.leak.tolist() if self.leak is not None else None,
            mode=self.mode,
            state_names=self.state_names.copy()
        )
        return new_cpd


def _is_stochastically_dominant(pmf_new: np.ndarray, pmf_old: np.ndarray, epsilon: float = 1e-9) -> bool:
    """
    Checks if pmf_new stochastically dominates pmf_old.
    (i.e., CDF_new(x) <= CDF_old(x) for all x)
    """
    cdf_new = np.cumsum(pmf_new)
    cdf_old = np.cumsum(pmf_old)
    return np.all(cdf_new <= (cdf_old + epsilon))

def _sample_dominant_fallback(pmf_old: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Guarantees a stochastically dominant sample using the
    formal (if "squashing") CDF sampling method.
    """
    child_card = len(pmf_old)
    last_cdf = np.cumsum(pmf_old)
    current_cdf = np.zeros(child_card)
    prev_val = 0.0 # F_j(x-1)
    
    for i in range(child_card - 1):
        upper_bound = last_cdf[i]
        lower_bound = prev_val
        
        if lower_bound > upper_bound:
            lower_bound = upper_bound
            
        val = rng.uniform(lower_bound, upper_bound)
        current_cdf[i] = val
        prev_val = val
    
    current_cdf[child_card - 1] = 1.0
    
    return np.diff(np.insert(current_cdf, 0, 0.0))

@staticmethod
def get_random_CI(
    variable: Hashable,
    evidence: List[Hashable],
    cardinality: Dict[Hashable, int],
    mode: str = 'or',
    isLeaky: bool = False,
    seed: Optional[int] = None,
    mass_shift_rate: float = 0.5,
    max_rejection_tries: int = 100,
) -> Union["BinaryInfluenceModel", "MultilevelInfluenceModel"]:
    """
    Creates a random CI model using Rejection Sampling to
    guarantee stochastic dominance for MIMs.
    """
    rng = np.random.default_rng(seed)
    mode_upper = mode.upper()
    
    child_card = cardinality[variable]
    parent_cards = [cardinality[p] for p in evidence]
    is_all_binary = (child_card == 2) and all(c == 2 for c in parent_cards)

    if mode_upper in {"OR", "AND"}:
        if not is_all_binary:
            raise ValueError(f"Mode '{mode_upper}' requires binary variables.")
        n_parents = len(evidence)
        activation_magnitude = rng.uniform(0.05, 0.95, size=n_parents)
        leak = rng.uniform(0.0, 0.1) if isLeaky else None
        
        return BinaryInfluenceModel(
            variable=variable,
            evidence=evidence,
            activation_magnitude=activation_magnitude,
            mode=mode_upper,
            leak=leak,
            isboolean_style=False
        )

    elif mode_upper in {"MAX", "MIN"}:
        
        influence_tables = {}
        state_names = {}
        state_names[variable] = list(range(child_card))

        for parent in evidence:
            parent_card = cardinality[parent]
            parent_degree = parent_card * child_card
            parent_states = list(range(parent_card))
            state_names[parent] = parent_states
            parent_table = {}
            
            last_pmf = np.zeros(child_card)
            last_pmf[0] = 1.0

            pmf_list = []

            current_alphas = (5/parent_card) * np.ones(child_card) / (1 + np.arange(1,child_card+1))
            current_alphas[0] = 5.0

            for state in parent_states:
                if state == 0:
                    current_pmf = last_pmf
                else:
                    next_alphas = current_alphas.copy()
                    for i in range(child_card - 1):
                        # Shift mass from bin i to bin i+1
                        delta = next_alphas[i] * mass_shift_rate * (child_card - i) / parent_degree
                        next_alphas[i] -= delta
                        next_alphas[i+1] += delta
                    
                    next_alphas[next_alphas < 0.1] = 0.1
                    current_alphas = next_alphas

                    found_sample = False
                    for _ in range(max_rejection_tries):
                        new_pmf = rng.dirichlet(current_alphas)
                        if _is_stochastically_dominant(new_pmf, last_pmf):
                            current_pmf = new_pmf
                            found_sample = True
                            break

                    if not found_sample:
                        current_pmf = _sample_dominant_fallback(last_pmf, rng)
                pmf_list.append(current_pmf)
                last_pmf = current_pmf
                
            if mode_upper=="MAX":
                parent_table = dict(zip(parent_states,pmf_list))
            else: #MIN case
                pmf_list.reverse()
                pmf_list_flip = map(np.flip, pmf_list)
                parent_table = dict(zip(parent_states,pmf_list_flip))
            influence_tables[parent] = parent_table
                

        leak_probs = None
        if isLeaky:
            alphas = np.ones(child_card)
            leak_probs = rng.dirichlet(alphas)

        return MultilevelInfluenceModel(
            variable=variable,
            evidence=evidence,
            influence_tables=influence_tables,
            levels=child_card,
            leak=leak_probs,
            mode=mode_upper,
            state_names=state_names,
        )
    else:
        raise ValueError(
            f"Unknown CI mode: '{mode}'. Must be one of 'OR', 'AND', 'MAX', 'MIN'."
        )


@staticmethod
def get_random_DBN(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: Optional[List[Hashable]] = None,
        n_states: Optional[Union[int, list[int], Dict[Hashable, int]]] = None,
        latents: bool = False,
        seed: Optional[int] = None,
        method: str = "erdos",
        **kwargs,
    ) -> "DiscreteBayesianNetwork":
        """
        Returns a randomly generated Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        n_states: int or dict (default: None)
            The number of states of each variable in the form
            {variable: no_of_states}. If a single value is provided,
            all nodes will have the same number of states. When None
            randomly generates the number of states.

        latents: bool (default: False)
            If True, also creates latent variables.

        seed: int (default: None)
            The seed value for random number generators.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork.get_random(n_nodes=5)
        >>> model.nodes()
        NodeView((0, 1, 3, 4, 2))
        >>> model.edges()
        OutEdgeView([(0, 1), (0, 3), (1, 3), (1, 4), (3, 4), (2, 3)])
        >>> model.cpds
        [<TabularCPD representing P(0:0) at 0x7f97e16eabe0>,
         <TabularCPD representing P(1:1 | 0:0) at 0x7f97e16ea670>,
         <TabularCPD representing P(3:3 | 0:0, 1:1, 2:2) at 0x7f97e16820d0>,
         <TabularCPD representing P(4:4 | 1:1, 3:3) at 0x7f97e16eae80>,
         <TabularCPD representing P(2:2) at 0x7f97e1682c40>]
        """
        gen = np.random.default_rng(seed=seed)
        if node_names is None:
            node_names = list([f"X_{i}" for i in range(n_nodes)])
 
        if n_states is None:
            n_states = gen.integers(low=1, high=5, size=n_nodes)
            n_states_dict = {node_names[i]: n_states[i] for i in range(n_nodes)}

        elif isinstance(n_states, int):
            n_states = np.array([n_states] * n_nodes)
            n_states_dict = {node_names[i]: n_states[i] for i in range(n_nodes)}

        elif isinstance(n_states, list):
            n_states = gen.choice(n_states, n_nodes)
            n_states_dict = {node_names[i]: n_states[i] for i in range(n_nodes)}

        elif isinstance(n_states, dict):
            n_states_dict = n_states

        dag = DAG.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            node_names=node_names,
            latents=latents,
            seed=seed,
            method = method,
            **kwargs,
        )
        bn_model = DiscreteBayesianNetwork(dag.edges(), latents=dag.latents)
        bn_model.add_nodes_from(dag.nodes())

        cpds = []
        for node in bn_model.nodes():
            parents = list(bn_model.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality=n_states_dict,
                    seed=seed,
                )
            )

        bn_model.add_cpds(*cpds)
        return bn_model

@staticmethod
def get_random_DAG(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: Optional[list[Hashable]] = None,
        latents: bool = False,
        seed: Optional[int] = None,
        method: str = "erdos",
        **kwargs,
    ) -> "DAG":
        """
        Returns a randomly generated DAG using different generation strategies.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes (used by some methods).

        node_names: list (default: None)
            A list of variable names to use in the random graph.
            If None, nodes are labeled X_0, X_1, ...

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        seed: int (default: None)
            Random seed for reproducibility.

        method: str
            Generation strategy. One of:
            {"erdos", "spanning_tree", "preferential", "layered"}.

        Additional kwargs
        -----------------
        For specific methods:
            - method="preferential": m (int, default=2) number of parents per new node.
            - method="layered": n_layers (int), layer_conn_prob (float)

        Returns
        -------
        Random DAG : pgmpy.base.DAG
        """
        rng = np.random.default_rng(seed)
        if node_names is None:
            node_names = [f"X_{i}" for i in range(n_nodes)]

        # ---- ERDŐS–RÉNYI STYLE DAG ----
        if method == "erdos":
            adj = rng.choice([0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob])
            adj = np.triu(adj, k=1)  # ensure acyclicity
            adj_pd = pd.DataFrame(adj, columns=node_names, index=node_names)
            nx_dag = nx.from_pandas_adjacency(adj_pd, create_using=nx.DiGraph)

        # ---- CONNECTED DAG VIA SPANNING TREE + NOISE ----
        elif method == "spanning_tree":
            order = rng.permutation(node_names)
            adj = np.zeros((n_nodes, n_nodes), dtype=int)

            # First, a directed spanning tree (acyclic)
            for i in range(1, n_nodes):
                parent = rng.integers(0, i)
                adj[parent, i] = 1  # edge along order

            # Add extra edges probabilistically (preserve DAG)
            extra = rng.random((n_nodes, n_nodes)) < edge_prob
            adj = np.triu(np.logical_or(adj, extra), 1).astype(int)

            adj_pd = pd.DataFrame(adj, columns=order, index=order)
            nx_dag = nx.from_pandas_adjacency(adj_pd, create_using=nx.DiGraph)

        # ---- DIRECTED PREFERENTIAL ATTACHMENT ----
        elif method == "preferential":
            m = kwargs.get("m", 2)
            G = nx.DiGraph()
            G.add_nodes_from(node_names)
            for new_idx in range(1, n_nodes):
                existing = list(range(new_idx))
                probs = np.array([G.in_degree(node_names[i]) + 1 for i in existing])
                probs = probs / probs.sum()
                n_parents = min(m, new_idx)
                parents = rng.choice(existing, size=n_parents, replace=False, p=probs)
                for p in parents:
                    G.add_edge(node_names[p], node_names[new_idx])
            nx_dag = G

        # ---- LAYERED DAG ----
        elif method == "layered":
            n_layers = kwargs.get("n_layers", int(np.sqrt(n_nodes)))
            layer_conn_prob = kwargs.get("layer_conn_prob", edge_prob)

            layers = [[] for _ in range(n_layers)]
            for node in node_names:
                l = rng.integers(0, n_layers)
                layers[l].append(node)

            G = nx.DiGraph()
            G.add_nodes_from(node_names)

            for i in range(n_layers - 1):
                src = layers[i]
                for j in range(i + 1, n_layers):
                    tgt = layers[j]
                    for u in src:
                        for v in tgt:
                            if rng.random() < layer_conn_prob:
                                G.add_edge(u, v)
            nx_dag = G

        else:
            raise ValueError(f"Unknown DAG generation method '{method}'")

        # ---- Add latent nodes optionally ----
        dag = DAG(nx_dag)
        if latents:
            n_latents = rng.integers(low=1, high=max(2, len(dag.nodes())))
            dag.latents = set(rng.choice(list(dag.nodes()), n_latents, replace=False))

        return dag



def to_nl_DBN(
        self, n_round: int = 4, minimalist: bool = False, random_minimalist: bool = True
    ) -> str:
        """
        Converts the entire Bayesian Network into a Natural Language description.
        
        It iterates over each node and calls its CPD's .to_NL() method.
        """
        full_description = []

        for cpd in self.get_cpds():
            node_nl_lines = cpd.to_nl(
                n_round=n_round,
                minimalist=minimalist,
                random_minimalist=random_minimalist
            )
            full_description.extend(node_nl_lines)
        
        return " \n".join(full_description)

def to_nl_CPD(self, n_round: int = 4, minimalist: bool = False, random_minimalist: bool = True) -> List[str]:
        """Converts TabularCPD to compact notation."""
        self.evidence = self.variables[1:] or None
        child_states = self.state_names[self.variable]
        probs_table = self.get_values()
        
        if not self.evidence:
            probs = {repr(s): float(round(probs_table[j, 0], n_round)) for j, s in enumerate(child_states)}
            return [f"P({self.variable}) = {probs}"]
        
        lines = []
        parent_combos = list(product(*[self.state_names[p] for p in self.evidence]))
        for i, combo in enumerate(parent_combos):
            cond = ", ".join(f"{p}={repr(v)}" for p, v in zip(self.evidence, combo))
            probs = {repr(s): float(round(probs_table[j, i], n_round)) for j, s in enumerate(child_states)}
            lines.append(f"P({self.variable}|{cond}) = {probs}")
        return lines

def to_nl_BIM(self, n_round: int = 4, **kwargs) -> List[str]: 
        """ 
        Converts the BinaryInfluenceModel into a more human-readable, conceptual 
        Natural Language description.
        """ 
        descriptions = [] 
        
        if self.mode.upper() == 'OR':
            model_name = "Noisy-OR"
            action_verb = "activate"
            base_action = "activation"
        elif self.mode.upper() == 'AND':
            model_name = "Noisy-AND"
            action_verb = "inhibit"
            base_action = "inhibition"
        else:
            return ValueError(f"""The mode of the interaction must be 'OR' or 'AND' but is instead {self.mode.upper()} """)

        active_state = repr(self.state_names[self.variable][1]) 
        
        base_desc = (
            f"The variable {self.variable} is "
            f"influenced by its parents: {', '.join(self.evidence)}."
        )
        descriptions.append(base_desc)

        concept_desc = (
            f"Each parent may activate independently {self.variable}, but they globally {action_verb} it."
        )
        descriptions.append(concept_desc)
        descriptions.append("The specific contributions are:")

        for parent, prob in zip(self.evidence, self.activation_magnitude): 
            parent_active = repr(self.state_names[parent][1]) 
            prob_rounded = round(prob, n_round) 
            
            descriptions.append( 
                f"  - When {parent} is active (is equal to {parent_active}), it has a "
                f"{prob_rounded} probability of successfully causing "
                f"the activation of {self.variable}."
            ) 

        if self.isleaky: 
            leak_prob = round(self.leak[0], n_round) 
            leak_desc_human = (
                f"  - There is also a base-line probability (or 'leak'). "
                f"If all parents are inactive, the probability of {self.variable} "
                f"being '{active_state}' is {leak_prob}."
            )
            descriptions.append(leak_desc_human)
            
        return descriptions


def to_nl_MIM(self, n_round: int = 4, **kwargs) -> List[str]:
        """
        Converts the MultilevelInfluenceModel into a sparse Natural Language description.
        """
        descriptions = []
        model_type = f"Noisy-{self.mode.upper()}"
        child_states = self.state_names[self.variable]
        
        base_desc = (
            f"The variable {self.variable} is the result of a {model_type} interaction due to its parents "
            f"({', '.join(self.evidence)})."
        )
        descriptions.append(base_desc)

        for parent in self.evidence:
            parent_states = self.state_names[parent]
            for state in parent_states:
                probs = self.influence_tables[parent][state]                    
                prob_parts = []
                for i, prob in enumerate(probs):
                    if prob > 1e-6:
                        prob_rounded = round(prob, n_round)
                        prob_parts.append(f"P({self.variable}={repr(child_states[i])})={prob_rounded}")
                
                prob_desc = ", ".join(prob_parts)
                descriptions.append(
                    f"  - The influence of {parent} = {repr(state)} (when others are inactive) is: [{prob_desc}]."
                )

        if self.isleaky:
            prob_parts = []
            for i, prob in enumerate(self.leak):
                if prob > 1e-6:
                    prob_rounded = round(prob, n_round)
                    prob_parts.append(f"P({self.variable}={repr(child_states[i])})={prob_rounded}")
            prob_desc = ", ".join(prob_parts)
            descriptions.append(
                f"  - The leak distribution (when all parents are inactive) is: [{prob_desc}]."
            )
        
        return descriptions

"""
def query_surgery(self,
                      variables,
                      do=None,
                      evidence=None,
                      show_progress=False,
                      **kwargs):

        # ---------- 1. Normalize inputs ----------
        if isinstance(variables, str):
            variables = [variables]
        if do is None:
            do = {}
        if evidence is None:
            evidence = {}

        # ---------- 2. No intervention → default inference ----------
        if do == {}:
            infer = VariableElimination(self.model)
            return infer.query(variables, evidence=evidence, show_progress=False)

        # ---------- 3. Graph surgery ----------
        # Build modified model M' where parents of do vars are removed
        model_prime = copy.deepcopy(self.model)

        # Remove parent edges → do(Y=y) surgery
        for y in do:
            for parent in list(model_prime.predecessors(y)):
                model_prime.remove_edge(parent, y)

    # ---------- 4. CPD Surgery with Silencing ----------
        # We temporarily disable WARNINGS to stop "Replacing existing CPD..."
        logging.disable(logging.WARNING)
        
        try:
            for d, val in do.items():
                state_names = self.model.get_cpds(d).state_names[d]
                k = len(state_names)
                probs = np.zeros(k)
                probs[state_names.index(val)] = 1.0
                values = np.array(probs).reshape(k, 1)

                cpd = TabularCPD(
                    variable=d,
                    variable_card=k,
                    values=values,
                    state_names={d: state_names}
                )

                # This line normally triggers the warning
                model_prime.add_cpds(cpd)
                
        finally:
            # Re-enable logging immediately after the loop
            logging.disable(logging.NOTSET)

        # ---------- 5. Use normal inference on the surgically modified model ----------
        infer = VariableElimination(model_prime)
        return infer.query(variables, evidence=evidence, show_progress=False)
"""

### BIF serialization ###

# --- 1. The Canonical BIF Writer ---
class CanonicalBIFWriter:
    def __init__(self, model):
        self.model = model

    def write_string(self) -> str:
        model = deepcopy(self.model)
        canonical_comments = []

        # Iterate over ALL nodes to capture types for everyone
        for node in list(model.nodes()):
            cpd = model.get_cpds(node)
            cls_name = cpd.__class__.__name__

            if cls_name in ["BinaryInfluenceModel", "MultilevelInfluenceModel"]:
                model.remove_cpds(cpd)
                model.add_cpds(self._dummy_tabular(cpd))
                canonical_comments.append(self._canonical_comment(cpd))
            
            # --- NEW: Save metadata for Standard TabularCPDs too ---
            elif cls_name == "TabularCPD":
                # We don't need to replace it with a dummy, just save the comment
                canonical_comments.append(self._canonical_comment(cpd))
            # -------------------------------------------------------

        try:
            bif = str(BIFWriter(model))
        except TypeError:
            writer = BIFWriter(model)
            bif = writer.get_string()

        return "\n".join(canonical_comments) + "\n\n" + bif

    def _dummy_tabular(self, cpd):
        card = cpd.variable_card
        if hasattr(cpd, "evidence_card"):
            evidence_card = cpd.evidence_card
        else:
            evidence_card = [len(cpd.state_names[parent]) for parent in cpd.evidence]
        
        n_cols = np.prod(evidence_card) if evidence_card else 1
        values = np.ones((card, int(n_cols))) / card
        
        return TabularCPD(
            variable=cpd.variable,
            variable_card=card,
            values=values,
            evidence=cpd.evidence,
            evidence_card=evidence_card,
            state_names=cpd.state_names
        )

    def _canonical_comment(self, cpd):
        lines = ["// CANONICAL", f"// variable: {cpd.variable}"]
        # Save exact state types for EVERYONE
        lines.append(f"// state_names: {cpd.state_names}") 
        
        if isinstance(cpd, BinaryInfluenceModel):
            activations = [float(x) for x in cpd.activation_magnitude]
            leak_val = float(cpd.leak.item()) if cpd.leak is not None else None
            
            lines.extend([
                "// type: BinaryInfluenceModel",
                f"// mode: {cpd.mode}",
                f"// leak: {leak_val}",
                f"// activation_magnitude: {activations}",
                f"// parents: {list(cpd.evidence)}",
            ])
            
        elif isinstance(cpd, MultilevelInfluenceModel):
            serializable_tables = {}
            for parent, table in cpd.influence_tables.items():
                serializable_tables[parent] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in table.items()
                }
            leak_val = cpd.leak.tolist() if cpd.leak is not None else None

            lines.extend([
                "// type: MultilevelInfluenceModel",
                f"// mode: {cpd.mode}",
                f"// leak: {leak_val}",
                f"// influence_tables: {serializable_tables}",
                f"// parents: {list(cpd.evidence)}",
            ])
            
        elif isinstance(cpd, TabularCPD):
            # Just tag it so the reader knows to restore state types
            lines.append("// type: TabularCPD")
            
        return "\n".join(lines)


# --- 2. The Canonical BIF Reader ---
class CanonicalBIFReader(BIFReader):
    def __init__(self, string=None, **kwargs):
        super().__init__(string=string, **kwargs)
        self._raw_text = string

    def get_model(self):
        model = super().get_model()

        for blob in self._parse_canonical_comments():
            variable = blob["variable"]
            c_type = blob.get("type")
            
            # If the blob has no 'parents' key, default to the model's structure
            parents = blob.get("parents", list(model.get_parents(variable)))

            # Get the state names from the comment (Correct Types)
            # Fallback to model if missing (Strings)
            if "state_names" in blob:
                state_names_map = blob["state_names"]
            else:
                state_names_map = {}
                child_cpd = model.get_cpds(variable)
                state_names_map[variable] = child_cpd.state_names[variable]
                for p in parents:
                    parent_cpd = model.get_cpds(p)
                    state_names_map[p] = parent_cpd.state_names[p]

            cpd = None

            if c_type == "BinaryInfluenceModel":
                cpd = BinaryInfluenceModel(
                    variable=variable,
                    evidence=parents,
                    activation_magnitude=blob["activation_magnitude"],
                    mode=blob["mode"],
                    leak=blob["leak"],
                    state_names=state_names_map 
                )
            elif c_type == "MultilevelInfluenceModel":
                cpd = MultilevelInfluenceModel(
                    variable=variable,
                    evidence=parents,
                    influence_tables=blob["influence_tables"],
                    levels=len(state_names_map[variable]),
                    leak=blob["leak"],
                    mode=blob["mode"],
                    state_names=state_names_map
                )
            elif c_type == "TabularCPD":
                existing_cpd = model.get_cpds(variable)
                
                # --- ROBUST EVIDENCE EXTRACTION ---
                # 'evidence' attribute might be missing.
                # In TabularCPD, variables[0] is the node, variables[1:] are parents.
                if hasattr(existing_cpd, "evidence"):
                    evidence = existing_cpd.evidence
                    evidence_card = existing_cpd.evidence_card
                else:
                    # Fallback: variables list is [child, parent1, parent2...]
                    # Note: pgmpy ensures 'variables' is always present.
                    evidence = existing_cpd.variables[1:]
                    evidence_card = existing_cpd.cardinality[1:]
                # ----------------------------------

                # Re-create TabularCPD to force the correct state types (Ints)
                cpd = TabularCPD(
                    variable=variable,
                    variable_card=existing_cpd.variable_card,
                    values=existing_cpd.get_values(),
                    evidence=evidence,
                    evidence_card=evidence_card,
                    state_names=state_names_map # <--- The Integers!
                )

            if cpd:
                model.remove_cpds(variable)
                model.add_cpds(cpd)

        return model

    def _parse_canonical_comments(self):
        blobs = []
        current = None
        if not self._raw_text: return blobs

        for line in self._raw_text.splitlines():
            line = line.strip()
            if line == "// CANONICAL":
                if current: blobs.append(current)
                current = {}
                continue
            if current is not None and line.startswith("//"):
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    try:
                        current[key] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        current[key] = val
                continue
            if current is not None and not line.startswith("//") and line:
                blobs.append(current)
                current = None
        if current: blobs.append(current)
        return blobs


# ==========================================
# 1. Semantic Tracker
#    Tracks probability formulas (Heads | Tails)
# ==========================================
# ==========================================
# 1. Semantic Tracker (Causal Aware)
# ==========================================
class SemanticTracker:
    def __init__(self, heads=None, tails=None, evidence_dict=None, do_dict=None):
        self.heads = set(heads) if heads else set()
        self.tails = set(tails) if tails else set()
        self.evidence = evidence_dict if evidence_dict else {}
        self.do = do_dict if do_dict else {}

    @classmethod
    def from_cpd(cls, cpd, evidence_dict, do_dict=None):
        heads = {cpd.variable}
        tails = set()
        relevant_ev = {}
        relevant_do = {}

        # 1. Check Heads
        if cpd.variable in evidence_dict:
            relevant_ev[cpd.variable] = evidence_dict[cpd.variable]
        if do_dict and cpd.variable in do_dict:
            relevant_do[cpd.variable] = do_dict[cpd.variable]

        # 2. Check Parents
        for parent in cpd.get_evidence():
            if parent in evidence_dict:
                relevant_ev[parent] = evidence_dict[parent]
            elif do_dict and parent in do_dict:
                relevant_do[parent] = do_dict[parent]
                # 'do' variables are technically tails (parents), just special ones
                tails.add(parent) 
            else:
                tails.add(parent)
                
        return cls(heads, tails, relevant_ev, relevant_do)

    def multiply(self, other):
        new_heads = self.heads.union(other.heads)
        raw_tails = self.tails.union(other.tails)
        new_tails = raw_tails - new_heads
        
        new_evidence = {**self.evidence, **other.evidence}
        new_do = {**self.do, **other.do}
        
        return SemanticTracker(new_heads, new_tails, new_evidence, new_do)

    def marginalize(self, variable):
        new_heads = self.heads - {variable}
        new_tails = self.tails.copy()
        return SemanticTracker(new_heads, new_tails, self.evidence, self.do)
    
    def project(self, active_variables):
        filtered_heads = self.heads.intersection(set(active_variables))
        return SemanticTracker(filtered_heads, self.tails, self.evidence, self.do)

    def __str__(self):
        if not self.heads:
            return "Evidence_Scalar"
        
        # 1. Format Heads
        head_parts = []
        observed_heads = set()
        
        for h in sorted(list(self.heads)):
            if h in self.evidence:
                head_parts.append(f"{h}={self.evidence[h]}")
                observed_heads.add(h)
            # If head is a 'do' variable (rare but possible in factor P(D)), show do(D)
            elif h in self.do:
                head_parts.append(f"do({h}={self.do[h]})")
                observed_heads.add(h)
            else:
                head_parts.append(h)
        
        h_str = ", ".join(head_parts)
        
        # 2. Format Tails
        conditions = set()
        
        # Process Tails
        for t in self.tails:
            if t in self.evidence:
                conditions.add(f"{t}={self.evidence[t]}")
            elif t in self.do:
                conditions.add(f"do({t}={self.do[t]})")
            else:
                conditions.add(t)
        
        # Process Remaining Context
        for var, val in self.evidence.items():
            if var not in observed_heads and var not in self.tails:
                conditions.add(f"{var}={val}")

        for var, val in self.do.items():
            if var not in observed_heads and var not in self.tails:
                 conditions.add(f"do({var}={val})")

        cond_list = sorted(list(conditions))
            
        if cond_list:
            return f"P({h_str} | {', '.join(cond_list)})"
        else:
            return f"P({h_str})"


# ==========================================
# 2. Base Semantic Inference Engine
# ==========================================
class SemanticTraceVE(VariableElimination):
    def __init__(self, model):
        super().__init__(model)
        self.trace_log = []
        self.factor_semantics = {}
        self.global_do = {} # New: store global intervention context

    def _format_value(self, val, scientific=False, precision=4):
        if scientific:
            fmt = f"{{:.{precision}e}}"
            s = fmt.format(val)
            base, exponent = s.split('e')
            if '.' in base: base = base.rstrip('0').rstrip('.')
            if exponent.startswith('+') or exponent.startswith('-'):
                sign = exponent[0]
                num = exponent[1:].lstrip('0')
                exponent = sign + (num if num else '0')
            return f"{base}e{exponent}"
        else:
            if float(val).is_integer(): return str(int(val))
            s = f"{{:.{precision}f}}".format(val)
            if '.' in s: s = s.rstrip('0').rstrip('.')
            return s

    def _render_factor_table(self, factor, semantics=None, scientific=False, precision=4):
        name = str(semantics) if semantics else "Unknown Factor"
        vars = factor.variables
        values = factor.values.flatten()
        card = factor.cardinality
        
        if not vars:
            val_str = self._format_value(values[0], scientific, precision)
            return f"Table for {name}:\n  [] = {val_str}"

        states = [range(c) for c in card]
        combinations = list(itertools.product(*states))
        lines = [f"Table for {name}:"]
        for i, comb in enumerate(combinations):
            assignments = ", ".join([f"{v}={s}" for v, s in zip(vars, comb)])
            val_str = self._format_value(values[i], scientific, precision)
            lines.append(f"  [{assignments}] = {val_str}")
        return "\n".join(lines)

    def _map_factors_to_cpds(self, working_factors, evidence):
        unique_factors = {}
        for bucket in working_factors.values():
            for factor, _ in bucket:
                unique_factors[id(factor)] = factor
        
        if not hasattr(self.model, 'cpds'): return 

        for cpd in self.model.cpds:
            temp_factor = cpd.to_factor()
            reduce_list = [(var, state) for var, state in evidence.items() if var in temp_factor.variables]
            if reduce_list:
                try: reduced_factor = temp_factor.reduce(reduce_list, inplace=False)
                except: continue
            else:
                reduced_factor = temp_factor

            for factor in unique_factors.values():
                if set(factor.variables) == set(reduced_factor.variables) and \
                   np.allclose(factor.values, reduced_factor.values):
                    # Pass the global 'do' context here
                    tracker = SemanticTracker.from_cpd(cpd, evidence, self.global_do)
                    self.factor_semantics[id(factor)] = tracker
                    
        for factor in unique_factors.values():
            if id(factor) not in self.factor_semantics:
                rel_ev = {k:v for k,v in evidence.items() if k in factor.variables}
                rel_do = {k:v for k,v in self.global_do.items() if k in factor.variables}
                self.factor_semantics[id(factor)] = SemanticTracker(
                    heads=set(factor.variables), 
                    evidence_dict=rel_ev,
                    do_dict=rel_do
                )

    def base_query(self, variables, evidence=None, virtual_evidence=None, elimination_order="MinFill", joint=True, show_progress=True):
        evidence = evidence if evidence is not None else dict()
        if isinstance(self.model, DiscreteBayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
        else:
            model_reduced = self.model

        reduced_ve = self.__class__(model_reduced)
        reduced_ve.trace_log = self.trace_log
        reduced_ve.factor_semantics = self.factor_semantics
        reduced_ve.global_do = self.global_do # Share Do Context
        reduced_ve._initialize_structures()

        return reduced_ve._variable_elimination(variables, "marginalize", evidence, elimination_order, joint, show_progress)

    def query_with_trace(self, variables, evidence=None, elimination_order="MinFill", joint=True):
        self.trace_log = []
        self.factor_semantics = {}
        result = self.base_query(variables, evidence, elimination_order=elimination_order, joint=joint, show_progress=False)
        return result, self.trace_log

    def _variable_elimination(self, variables, operation, evidence=None, elimination_order="MinFill", joint=True, show_progress=True):
        if not variables:
            return factor_product(*self.factors.values()) if joint else set(self.factors.values())
        
        eliminated_variables = set()
        working_factors = self._get_working_factors(evidence)
        elimination_order = self._get_elimination_order(variables, evidence, elimination_order, show_progress)
        
        self._map_factors_to_cpds(working_factors, evidence)
        self.trace_log.append({"step": "INIT", "elimination_order": elimination_order})

        for var in elimination_order:
            factors = [f for f, _ in working_factors[var] if not set(f.variables).intersection(eliminated_variables)]
            
            if not factors:
                del working_factors[var]
                eliminated_variables.add(var)
                continue

            input_data = []
            for f in factors:
                tracker = self.factor_semantics.get(id(f))
                input_data.append((f, tracker))

            phi = factor_product(*factors)
            phi_tracker = input_data[0][1]
            for _, next_tracker in input_data[1:]:
                phi_tracker = phi_tracker.multiply(next_tracker)
            self.factor_semantics[id(phi)] = phi_tracker
            
            product_expr = " * ".join([str(t) for _, t in input_data])
            product_formula = str(phi_tracker)

            tau = getattr(phi, operation)([var], inplace=False)
            tau_tracker = phi_tracker.marginalize(var)
            self.factor_semantics[id(tau)] = tau_tracker
            sum_formula = str(tau_tracker)

            self.trace_log.append({
                "step": "ELIMINATE",
                "variable": var,
                "input_data": input_data,
                "product_data": (phi, phi_tracker),
                "sum_data": (tau, tau_tracker),
                "product_expr": product_expr,
                "product_formula": product_formula,
                "sum_formula": sum_formula
            })

            del working_factors[var]
            for variable in phi.variables:
                 if variable in working_factors:
                    working_factors[variable].add((tau, var))
            for variable in tau.variables:
                if variable not in working_factors:
                    working_factors[variable] = set()
                working_factors[variable].add((tau, var))
            eliminated_variables.add(var)

        final_factors_map = {}
        for bucket in working_factors.values():
            for factor, _ in bucket:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_factors_map[id(factor)] = factor
        
        unique_factors = list(final_factors_map.values())
        
        final_input_data = []
        for f in unique_factors:
            tracker = self.factor_semantics.get(id(f))
            final_input_data.append((f, tracker))

        if unique_factors:
            final_tracker = self.factor_semantics[id(unique_factors[0])]
            for f in unique_factors[1:]:
                final_tracker = final_tracker.multiply(self.factor_semantics[id(f)])
            
            unnormalized = factor_product(*unique_factors)
            clean_tracker = final_tracker.project(unnormalized.variables)
            
            if joint:
                normalized = unnormalized.normalize(inplace=False)
                self.trace_log.append({
                    "step": "NORMALIZE",
                    "final_input_data": final_input_data,
                    "unnormalized_data": (unnormalized, clean_tracker),
                    "normalized_data": (normalized, clean_tracker),
                    "final_formula": str(clean_tracker)
                })
                return normalized
            else:
                return unnormalized
        else:
             return DiscreteFactor(variables=[], cardinality=[], values=[1.0])

    def generate_natural_language_proof(self, scientific_notation=False, precision=2, concise=True):
        """Generate proof. If concise=True, use compact token-efficient format."""
        if concise:
            return self._generate_concise_proof(scientific_notation, precision)
        return self._generate_verbose_proof(scientific_notation, precision)
    
    def _render_inline_dist(self, factor, precision=2):
        """Render distribution as inline dict: {0: 0.65, 1: 0.35}"""
        vars = factor.variables
        values = factor.values.flatten()
        if len(vars) == 1:
            card = factor.cardinality[0]
            return "{" + ", ".join(f"{i}: {round(values[i], precision)}" for i in range(card)) + "}"
        # Multi-var: use compact tuple notation
        card = factor.cardinality
        states = [range(c) for c in card]
        combos = list(itertools.product(*states))
        parts = [f"{comb}: {round(values[i], precision)}" for i, comb in enumerate(combos)]
        return "{" + ", ".join(parts) + "}"
    
    def _generate_concise_proof(self, scientific, precision):
        """Token-efficient proof format for LLM pre-training."""
        lines = []
        
        for entry in self.trace_log:
            step = entry['step']
            
            if step == 'GOAL':
                lines.append(f"Goal: {entry['description']}")
            
            elif step == 'SURGERY':
                surgery_parts = []
                for d in entry['details']:
                    if "Point Mass" in d:
                        surgery_parts.append(d.replace("Set P(", "P(").replace(") :=", ")="))
                    elif "no parents" not in d:
                        surgery_parts.append(d)
                if surgery_parts:
                    lines.append("Surgery: " + "; ".join(surgery_parts))
            
            elif step == 'INIT':
                order = entry.get('elimination_order', [])
                if order:
                    lines.append(f"Elim order: {order}")
            
            elif step == 'ELIMINATE':
                var = entry['variable']
                tau, tau_track = entry['sum_data']
                result_str = self._render_inline_dist(tau, precision)
                lines.append(f"Sum out {var} -> {entry['sum_formula']} = {result_str}")
            
            elif step == 'TRIVIAL_DO':
                do_var, do_val, target = entry['do_var'], entry['do_val'], entry['target']
                result = entry['result']
                result_str = self._render_inline_dist(result, precision)
                lines.append(f"do({do_var}={do_val}) trivial -> P({target}) = {result_str}")
            
            elif step == 'NORMALIZE':
                n_factor, n_track = entry['normalized_data']
                u_factor, _ = entry['unnormalized_data']
                u_sum = u_factor.values.sum()
                result_str = self._render_inline_dist(n_factor, precision)
                if abs(u_sum - 1.0) < 1e-6:
                    lines.append(f"Result: {entry['final_formula']} = {result_str}")
                else:
                    lines.append(f"Normalize (sum={round(u_sum, precision)}) -> {entry['final_formula']} = {result_str}")
        
        return "\n".join(lines)
    
    def _generate_verbose_proof(self, scientific_notation, precision):
        """Original verbose proof format with full tables."""
        proof = []
        has_eliminations = any(e['step'] == 'ELIMINATE' for e in self.trace_log)
        for entry in self.trace_log:
            step = entry['step']
            if step == 'INIT' and not entry.get('elimination_order'):
                continue
            if step == 'NORMALIZE' and not has_eliminations:
                final_input = entry.get('final_input_data', [])
                n_factor, n_track = entry['normalized_data']
                if len(final_input) > 1:
                    factors_str = " * ".join(str(t) for _, t in final_input if t)
                    proof.append(f"Combining factors: {factors_str}")
                table_str = self._render_factor_table(n_factor, n_track, scientific_notation, precision)
                proof.append(f"Result:\n{table_str}")
                continue
            text = self._render_trace_entry(entry, scientific_notation, precision)
            if text:
                proof.append(text)
        return "\n".join(proof)

    def _render_trace_entry(self, entry, scientific, precision):
        """
        Renders a single trace entry. 
        Can be overridden by subclasses to handle specific step types.
        """
        step = entry['step']
        
        if step == 'GOAL':
            return f"Goal: {entry['description']}\n"
        
        elif step == 'INIT':
            return f"Initialization: Selected Elimination Order = {entry['elimination_order']}\n"
        
        elif step == 'ELIMINATE':
            var = entry['variable']
            lines = [f"--- Step: Eliminate Variable '{var}' ---"]
            
            lines.append(f"1. Retrieve relevant factors containing '{var}':")
            for (factor, tracker) in entry['input_data']:
                table_str = self._render_factor_table(factor, tracker, scientific, precision)
                lines.append("\n".join(["   " + line for line in table_str.split('\n')]))
            
            lines.append(f"\n2. Compute the Intermediate Joint (Product):")
            lines.append(f"   Formula: {entry['product_formula']} = {entry['product_expr']}")
            phi, phi_track = entry['product_data']
            table_str = self._render_factor_table(phi, phi_track, scientific, precision)
            lines.append("\n".join(["   " + line for line in table_str.split('\n')]))
            
            lines.append(f"\n3. Marginalize (Sum) out variable '{var}':")
            lines.append(f"   Formula: {entry['sum_formula']} = \u2211_{{{var}}} {entry['product_formula']}")
            tau, tau_track = entry['sum_data']
            table_str = self._render_factor_table(tau, tau_track, scientific, precision)
            lines.append("\n".join(["   " + line for line in table_str.split('\n')]))
            lines.append("\n")
            return "\n".join(lines)

        elif step == 'NORMALIZE':
            lines = [f"--- Final Step: Normalization ---"]
            
            lines.append(f"1. Gather all remaining factors (Query Variables + Priors):")
            for (factor, tracker) in entry['final_input_data']:
                table_str = self._render_factor_table(factor, tracker, scientific, precision)
                lines.append("\n".join(["   " + line for line in table_str.split('\n')]))

            lines.append(f"\n2. Compute Unnormalized Joint Distribution ({entry['final_formula']}):")
            u_factor, u_track = entry['unnormalized_data']
            table_str = self._render_factor_table(u_factor, u_track, scientific, precision)
            lines.append("\n".join(["   " + line for line in table_str.split('\n')]))
            
            lines.append(f"\n3. Normalize to obtain Probability Distribution:")
            n_factor, n_track = entry['normalized_data']
            table_str = self._render_factor_table(n_factor, n_track, scientific, precision)
            lines.append("\n".join(["   " + line for line in table_str.split('\n')]))
            return "\n".join(lines)
        
        return None


# ==========================================
# 3. Optimized Causal Inference Engine
# ==========================================
class CausalVE(SemanticTraceVE):
    def query(self, variables, do=None, evidence=None, elimination_order="MinFill", show_progress=False):
        if do is None: do = {}
        if evidence is None: evidence = {}
        
        self.trace_log = []
        self.factor_semantics = {}
        self.global_do = do 
        
        # 1. Format the Goal
        do_str = ", ".join([f"do({k}={v})" for k, v in do.items()])
        ev_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
        query_parts = [p for p in [do_str, ev_str] if p]
        context = f" | {', '.join(query_parts)}" if query_parts else ""
        query_str = f"P({', '.join(variables)}{context})"
        
        self.trace_log.append({
            "step": "GOAL",
            "description": f"Compute {'Causal Effect' if do else 'Observational Probability'}: {query_str}"
        })

        # --- OPTIMIZATION: If no DO variables, skip surgery entirely ---
        if not do:
            return self.base_query(variables, evidence, elimination_order=elimination_order, joint=True, show_progress=False)

        # --- OPTIMIZATION: Trivial causal case (do-var has no parents, target is direct child) ---
        if len(do) == 1 and len(variables) == 1:
            do_var, do_val = list(do.items())[0]
            target = variables[0]
            if not list(self.model.predecessors(do_var)) and do_var in self.model.predecessors(target):
                # Just lookup from CPT: P(target | do(do_var=val)) = P(target | do_var=val)
                cpd = self.model.get_cpds(target)
                combined_ev = {**evidence, do_var: do_val}
                result = cpd.to_factor().reduce([(k, v) for k, v in combined_ev.items()], inplace=False)
                result = result.normalize(inplace=False)
                self.trace_log.append({
                    "step": "TRIVIAL_DO",
                    "do_var": do_var, "do_val": do_val, "target": target,
                    "result": result
                })
                return result

        # --- Surgery (Only if do exists) ---
        model_prime = copy.deepcopy(self.model)
        surgery_details = []

        for y in do:
            predecessors = list(model_prime.predecessors(y))
            if predecessors:
                surgery_details.append(f"Cut incoming edges to intervened node '{y}': {predecessors} -> {y}")
                for parent in predecessors:
                    model_prime.remove_edge(parent, y)
            else:
                surgery_details.append(f"Node '{y}' has no parents to cut.")

        logging.disable(logging.WARNING) 
        try:
            for d, val in do.items():
                cpd_old = model_prime.get_cpds(d)
                if cpd_old:
                    model_prime.remove_cpds(cpd_old)
                
                state_names = cpd_old.state_names[d]
                k = len(state_names)
                probs = np.zeros(k)
                idx = state_names.index(val) if val in state_names else int(val)
                probs[idx] = 1.0 
                values = np.array(probs).reshape(k, 1)

                new_cpd = TabularCPD(variable=d, variable_card=k, values=values, state_names={d: state_names})
                model_prime.add_cpds(new_cpd)
                surgery_details.append(f"Set P({d}) := Point Mass at {d}={val}.")
        finally:
            logging.disable(logging.NOTSET)

        self.trace_log.append({"step": "SURGERY", "details": surgery_details})

        # --- Fix Elimination Order for Surgery Graph ---
        final_order = elimination_order
        if isinstance(elimination_order, list):
            missing_vars = [
                node for node in model_prime.nodes() 
                if node not in variables 
                and node not in evidence 
                and node not in elimination_order
            ]
            if missing_vars:
                final_order = elimination_order + missing_vars

        # --- Run Inference on Mutilated Graph ---
        inference_prime = CausalVE(model_prime)
        inference_prime.global_do = self.global_do 
        
        result, prime_trace = inference_prime.query_with_trace(
            variables=variables,
            evidence=evidence, 
            elimination_order=final_order, 
            joint=True
        )

        self.trace_log.extend(prime_trace)
        return result

    def _render_trace_entry(self, entry, scientific, precision):
        """Override to handle SURGERY and TRIVIAL_DO."""
        if entry['step'] == 'TRIVIAL_DO':
            do_var, do_val, target = entry['do_var'], entry['do_val'], entry['target']
            result = entry['result']
            table_str = self._render_factor_table(result, None, scientific, precision)
            return f"Since {do_var} has no parents, do({do_var}={do_val}) = P({target}|{do_var}={do_val})\nResult:\n{table_str}"
        
        if entry['step'] == 'SURGERY':
            lines = ["--- Causal Graph Surgery ---"]
            for detail in entry['details']:
                lines.append(f"  - {detail}")
            lines.append("")
            return "\n".join(lines)
        
        return super()._render_trace_entry(entry, scientific, precision)