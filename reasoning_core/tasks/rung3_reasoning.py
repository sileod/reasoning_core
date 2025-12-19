import networkx as nx
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Any
from abc import ABC, abstractmethod

# We still use scmodels to hold the formal definition
from scmodels import SCM
import scmodels.parser

# --- Monkey Patching for Safety ---
# Ensure 'Normal' is supported (it usually is, but good practice)
if "Normal" not in scmodels.parser.all_stats_imports:
    scmodels.parser.all_stats_imports.add("Normal")

from reasoning_core.template import Task, Problem, Config

# --- 1. Configuration ---

@dataclass
class Rung3FloatConfig(Config):
    """
    Configuration for Continuous Counterfactual tasks.
    """
    n_nodes: int = 3
    edge_prob: float = 0.4
    # We don't need domain_size anymore.
    # We add precision to keep numbers clean.
    precision: int = 1 
    graph_generation_mode: str = "erdos"

    def update(self, c):
        self.n_nodes += c

# --- 2. The Logic Class ---

class BaseSCMGraph(ABC):
    """
    Abstract Base Class for Structural Causal Models (Rung 3).
    Handles DAG generation, Counterfactual logic, and NL formatting.
    """
    def __init__(self):
        self.scm = None
        self.dag = None
        self.equations_nl = {} 
        self.observed_values = {}
        self.noise_values = {}   
        self.counterfactual_query = {} 
        self.target_node = None
        self.truth_value = 0.0
        self.precision = 2
        self.model_name = "Structural Causal Model"
        self.model_description = "Variables are continuous."
        self.noise_separation = 2.5
        self.noise_scale = 1.0

    def _generate_dag_structure(self, n, edge_prob, seed):
        """Generates a random DAG and ensures topological order is respected."""
        rng = np.random.default_rng(seed)
        G = nx.gnp_random_graph(n, edge_prob, directed=True, seed=seed)
        # Enforce Acyclicity (u < v)
        self.dag = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
        self.dag.add_nodes_from(range(n))
        
        # Return sorted nodes for safe processing
        try:
            return list(nx.topological_sort(self.dag))
        except:
            return sorted(list(self.dag.nodes()))

    @abstractmethod
    def generate_random_scm(self, n, edge_prob, precision, seed):
        """Define equations and build SCM."""
        pass

    @abstractmethod
    def _manual_forward_pass(self, noise_dict, intervention_dict=None):
        """Calculate variable values from noise + parents."""
        pass

    def generate_counterfactual_problem(self,):
        """
        The Universal Algorithm for Counterfactuals (Rung 3).
        1. Generate Noise (N).
        2. Calculate Fact (X) from N.
        3. Define Intervention (do(X_i)).
        4. Calculate Counterfactual (X') from N + do(X_i).
        """
        rng = np.random.default_rng()
        
        # 1. Generate Hidden Noise
        self.noise_values = {}
        for node in self.dag.nodes():
            raw_noise = float(rng.standard_normal())
            self.noise_values[f"N_{node}"] = round(raw_noise, self.precision)
            
        # 2. Forward Pass (The Fact)
        raw_facts = self._manual_forward_pass(self.noise_values)
        self.observed_values = {f"X_{k}": v for k, v in raw_facts.items()}
        
        # 3. Define Intervention
        # Pick a target (sink) and an intervention node (upstream)
        try:
            ordered_nodes = list(nx.topological_sort(self.dag))
        except:
            ordered_nodes = sorted(list(self.dag.nodes()))

        self.target_node = f"X_{ordered_nodes[-1]}"
        upstream = ordered_nodes[:-1]
        
        if not upstream: return # Degenerate case

        intervene_node = random.choice(upstream)
        intervene_var = f"X_{intervene_node}"
        
        # Choose a value distinct from the observed fact
        curr_val = self.observed_values[intervene_var]

        chi_sample = np.sqrt(rng.chisquare(df=3))
        sign = rng.choice([-1, 1])
        
        shift = float(sign * chi_sample)
        new_val = round(curr_val + shift, self.precision)

        self.counterfactual_query = {intervene_var: new_val}
        
        # 4. Prediction (The Counterfactual)
        # Critical: We reuse self.noise_values (The Abduction step is implicit here)
        cf_values = self._manual_forward_pass(self.noise_values, self.counterfactual_query)
        self.truth_value = cf_values[int(self.target_node.split("_")[1])]

    def to_NL_description(self):
        lines = [f"### {self.model_name}"]
        lines.append(self.model_description)
        lines.append("Equations:")
        for node in sorted(self.dag.nodes()):
            lines.append(f"- {self.equations_nl[node]}")
        return "\n".join(lines)

    def to_NL_scenario(self):
        facts = ", ".join([f"{k}={v}" for k, v in self.observed_values.items()])
        q_var = list(self.counterfactual_query.keys())[0]
        q_val = list(self.counterfactual_query.values())[0]
        return (
            f"1. OBSERVED DATA: {facts}\n"
            f"2. QUERY: What if {q_var} had been {q_val}?"
        )

class LinearSCMGraph(BaseSCMGraph):
    def __init__(self):
        super().__init__()
        self.model_name = "Linear Causal Model"
        self.model_description = "All relationships are linear sums: Y = w1*X1 + ... + N."
        self.coefficients = {}

    def generate_random_scm(self, n=3, edge_prob=0.4, precision=1, seed=None):
        self.precision = precision
        ordered_nodes = self._generate_dag_structure(n, edge_prob, seed)
        rng = np.random.default_rng(seed)
        
        conf = []
        self.equations_nl = {}
        self.coefficients = {}

        for node in ordered_nodes:
            parents = list(self.dag.predecessors(node))
            noise_name = f"N_{node}"
            
            weights = {}
            eq_parts = []
            nl_parts = []
            
            for p in parents:
                w = round(rng.uniform(-1.5, 1.5), self.precision)
                if w == 0: w = 0.5 
                weights[p] = w
                eq_parts.append(f"{w} * X_{p}")
                nl_parts.append(f"{w} * X_{p}")
            
            self.coefficients[node] = weights

            if not parents:
                eq_str = noise_name
                nl_desc = f"X_{node} = {noise_name}"
            else:
                eq_str = " + ".join(eq_parts) + f" + {noise_name}"
                nl_desc = f"X_{node} = {' + '.join(nl_parts)} + {noise_name}"

            conf.append(f"X_{node} = {eq_str}; {noise_name} ~ Normal(0, 1)")
            self.equations_nl[node] = nl_desc

        self.scm = SCM(conf)

    def _manual_forward_pass(self, noise_dict, intervention_dict=None):
        values = {}
        # Simple Linear logic
        for node in sorted(list(self.dag.nodes())): # Topological sort implied by generation order usually, but safer to re-sort if needed.
            if intervention_dict and f"X_{node}" in intervention_dict:
                values[node] = intervention_dict[f"X_{node}"]
                continue
                
            val = 0.0
            for p in self.dag.predecessors(node):
                # Ensure parent is already computed. 
                # (Strictly, this loop relies on the node order being topological)
                if p in values:
                    val += self.coefficients[node][p] * values[p]
            
            val += noise_dict[f"N_{node}"]
            values[node] = round(val, self.precision)
            
        return values

class ComplexSCMGraph(BaseSCMGraph):
    def __init__(self):
        super().__init__()
        self.model_name = "Complex Causal Model"
        self.model_description = "Relationships may be Linear (sums), Quadratic (squared), or Interactions (products)."
        self.node_configs = {}

    def generate_random_scm(self, n=3, edge_prob=0.4, precision=1, seed=None):
        self.precision = precision
        ordered_nodes = self._generate_dag_structure(n, edge_prob, seed)
        rng = np.random.default_rng(seed)
        
        conf = []
        self.equations_nl = {}
        self.node_configs = {}

        for node in ordered_nodes:
            parents = list(self.dag.predecessors(node))
            noise_name = f"N_{node}"
            
            # Decide Type
            eq_type = "linear"
            if len(parents) >= 2 and rng.random() < 0.3: eq_type = "interaction"
            elif len(parents) == 1 and rng.random() < 0.3: eq_type = "quadratic"

            weights = {}
            for p in parents:
                w = round(rng.uniform(-1.0, 1.0), self.precision)
                if w == 0: w = 0.5
                weights[p] = w

            # Construct String
            if not parents:
                eq_str = noise_name
                nl_desc = f"X_{node} = {noise_name}"
            elif eq_type == "linear":
                terms = [f"{weights[p]} * X_{p}" for p in parents]
                eq_str = " + ".join(terms) + f" + {noise_name}"
                nl_desc = f"X_{node} = {' + '.join(terms)} + {noise_name}"
            elif eq_type == "interaction":
                prod = " * ".join([f"X_{p}" for p in parents])
                eq_str = f"{prod} + {noise_name}"
                nl_desc = f"X_{node} = ({prod}) + {noise_name}"
            elif eq_type == "quadratic":
                p = parents[0]
                eq_str = f"{weights[p]} * X_{p}**2 + {noise_name}"
                nl_desc = f"X_{node} = {weights[p]} * X_{p}^2 + {noise_name}"

            self.node_configs[node] = {'type': eq_type, 'weights': weights, 'parents': parents}
            conf.append(f"X_{node} = {eq_str}; {noise_name} ~ Normal(0, 1)")
            self.equations_nl[node] = nl_desc

        self.scm = SCM(conf)

    def _manual_forward_pass(self, noise_dict, intervention_dict=None):
        values = {}
        # Complex Logic
        try:
            nodes = list(nx.topological_sort(self.dag))
        except:
            nodes = sorted(list(self.dag.nodes()))

        for node in nodes:
            if intervention_dict and f"X_{node}" in intervention_dict:
                values[node] = intervention_dict[f"X_{node}"]
                continue
            
            cfg = self.node_configs[node]
            val = 0.0
            
            if cfg['parents']:
                if cfg['type'] == 'linear':
                    for p in cfg['parents']: val += cfg['weights'][p] * values[p]
                elif cfg['type'] == 'interaction':
                    prod = 1.0
                    for p in cfg['parents']: prod *= values[p]
                    val = prod
                elif cfg['type'] == 'quadratic':
                    p = cfg['parents'][0]
                    val = cfg['weights'][p] * (values[p]**2)
            
            val += noise_dict[f"N_{node}"]
            values[node] = round(val, self.precision)
            
        return values

# --- 3. The Rung 3 Task ---

class FloatCounterfactual(Task):
    def __init__(self, config=Rung3FloatConfig(), mode="complex"):
        super().__init__(config=config)
        
        if mode == "linear":
            self.scm_graph = LinearSCMGraph()
        else:
            self.scm_graph = ComplexSCMGraph()

    def _generate_network(self, n, edge_prob, method="erdos", **kwargs):
        # Map config to generator
        self.scm_graph.generate_random_scm(
            n=n, 
            edge_prob=edge_prob, 
            precision=self.config.precision
        )

    def _generate_specific_problem(self):
        self.scm_graph.generate_counterfactual_problem()

    def _calculate_answer_and_metadata(self):
        # The answer is a single float value
        true_val = self.scm_graph.truth_value
        
        # For the dictionary format required by the evaluator:
        # Since it's continuous, we can't return a full PDF.
        # We return the point estimate as a dict with probability 1.0
        # This is compatible with your scoring if the prediction is exact.
        return (true_val, {"exact_value": true_val})

    def _construct_scenario(self):
        return self.scm_graph.to_NL_scenario()

    def generate(self):
        self._generate_network(
            n=self.config.n_nodes,
            edge_prob=self.config.edge_prob
        )
        
        # Retry loop to ensure valid non-trivial graph
        for _ in range(10):
            try:
                self._generate_specific_problem()
                if self.scm_graph.target_node:
                    break
            except:
                continue
                
        answer, meta = self._calculate_answer_and_metadata()
        
        problem_data = {
            "target_var_values": "Continuous (Float)",
            "system_description": self.scm_graph.to_NL_description(),
            "scenario": self.scm_graph.to_NL_scenario(),
            "target": self.scm_graph.target_node,
            "variables": [f"X_{n}" for n in self.scm_graph.dag.nodes()]
        }
        problem_data.update(meta)
        
        return Problem(metadata=problem_data, answer=answer)

    def prompt(self, metadata):
        sys_desc = metadata['system_description']
        scenario = metadata['scenario']
        target = metadata['target']
        
        return (
            f"### System Instructions\n"
            f"You are an expert in Causal Inference performing Counterfactual Analysis.\n"
            f"Use the 'Abduction-Action-Prediction' method:\n"
            f"1. ABDUCTION: Use the Observed Data and Equations to calculate the hidden noise terms (N).\n"
            f"2. PREDICTION: Use those specific N values with the new Intervention value to predict the target.\n\n"
            f"{sys_desc}\n\n"
            f"### The Problem\n"
            f"{scenario}\n"
            f"Calculate the counterfactual value of **{target}**."
        )

from template import _REGISTRY
if 'float_counterfactual' in _REGISTRY:
    del _REGISTRY['float_counterfactual']