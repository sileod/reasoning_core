import subprocess
import re
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, field
from reasoning_core.utils.udocker_process import prover_session
import traceback

@dataclass
class DerivationNode:
    """Node representing a clause in the E-prover derivation graph."""
    
    clause_id: str
    clause_formula: str
    parents: List[str] = field(default_factory=list)
    inference: str = ""
    role: str = "plain"
    interesting_score: float = 0.0      
    other_agint_metrics: Dict[str, float] = field(default_factory=dict)
    full_cnf_clause: str = ""


def generate_derivation_graph(axiom_file: str, save_output: bool = True,
                                output_dir: str = "eprover_output",ranking: bool = True,
                                e_limit : int = 1):
    """
    Generates the derivation graph by running E-prover and optionally enriching it with AGInT scores.
    """
    G = run_eprover_and_build_graph(axiom_file,save_output,output_dir,e_limit)
    if ranking == True :
        G = enrich_graph_with_agint(G)
    return G

def run_eprover_and_build_graph(axiom_file: str, save_output: bool = True, 
                                output_dir: str = "eprover_output",e_limit : int = 2) -> nx.DiGraph:
    """
    Runs E-prover and builds the derivation graph from its output.
    """
    
    # 1. Run E-prover with udocker
    cmd = [
        '--proof-graph=2',
        '--full-deriv',
        '--force-deriv=1',
        '--output-level=1',
        f'--soft-cpu-limit={e_limit}',
    ]
    
    try:
        result = prover_session.run_prover('eprover',cmd,axiom_file)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("E-prover timeout")
        return nx.DiGraph()
    except Exception as e:
        print(f"Error E-prover: {type(e).__name__}: {e}")
        traceback.print_exc()
        return nx.DiGraph()
    
    # 2. Save output if requested
    if save_output:
        Path(output_dir).mkdir(exist_ok=True)
        base_name = Path(axiom_file).stem
        with open(f"{output_dir}/{base_name}_stdout.txt", 'w') as f:
            f.write(stdout)
        with open(f"{output_dir}/{base_name}_stderr.txt", 'w') as f:
            f.write(stderr)
    
    # 3. Parse and build the graph
    return _parse_digraph_to_networkx(stdout)


def _parse_digraph_to_networkx(eprover_output: str) -> nx.DiGraph:
    """Parses the E-prover digraph and returns a NetworkX graph."""
    
    # Extract the digraph content
    start = eprover_output.find("digraph proof{")
    if start == -1:
        return nx.DiGraph()
    
    end = eprover_output.rfind("}")
    digraph_content = eprover_output[start:end + 1]
    
    # Parse nodes and edges
    nodes = _extract_nodes(digraph_content)
    edges = _extract_edges(digraph_content)
    
    # Build the NetworkX graph
    graph = nx.DiGraph()
    
    # Add nodes
    for node_id, node_obj in nodes.items():
        graph.add_node(node_id, data=node_obj)

    # Add edges and update parent lists
    for parent, child in edges:
        if parent in nodes and child in nodes:
            graph.add_edge(parent, child)
            nodes[child].parents.append(parent)
    
    return graph


def _extract_nodes(digraph: str) -> Dict[str, 'DerivationNode']:
    """Extracts all nodes from the digraph string."""
    
    nodes = {}
    
    node_pattern = r'^\s*(\d+)\s*\[[^\]]*?label="((?:.|\n)*?)"'
    
    for match in re.finditer(node_pattern, digraph, re.MULTILINE):

        node_num = match.group(1)
    
        label_content = match.group(2).replace('\\n', '\n')
        
        clause_id, role, formula, inference = _parse_node_label(label_content)
        
        if clause_id:
            if node_num not in nodes:
                node = DerivationNode(
                    clause_id=clause_id,
                    clause_formula=formula,
                    role=role,
                    inference=inference,
                    full_cnf_clause=f"cnf({clause_id},{role},{formula})"
                )
                nodes[node_num] = node
    return nodes


def _extract_edges(digraph: str) -> list:
    """Extracts all edges from the digraph string."""
    
    edges = []
    edge_pattern = r'(\d+)\s*->\s*(\d+)'
    
    for match in re.finditer(edge_pattern, digraph):
        parent = match.group(1)
        child = match.group(2)
        edges.append((parent, child))
    
    return edges


def _parse_node_label(label: str) -> Tuple[str, str, str, str]:
   
    """ Parses a node label to extract clause_id, role, formula, and inference. """

    # Clean the label string
    clean_label = label.strip()
    
    # Extract the cnf(...) part
    cnf_match = re.match(r'cnf\(([^,]+),\s*([^,]+),\s*(.+)', clean_label, re.DOTALL)
    if not cnf_match:
        return "", "", "", ""
    
    clause_id = cnf_match.group(1).strip()
    role = cnf_match.group(2).strip()
    rest = cnf_match.group(3).strip()
    
    # Separate the formula and inference parts
    # Search for the first occurrence of a newline followed by a keyword
    inference_start = re.search(r'\n(inference|file|[a-z_]+)', rest)
    
    if inference_start:
        formula = rest[:inference_start.start()].strip()
        inference = rest[inference_start.start():].strip()
    else:
        formula = rest
        inference = ""
    
    # Clean up the trailing comma if present
    if formula.endswith(','):
        formula = formula[:-1]
    
    formula = formula.strip()
    if inference.startswith('inference'):
        inference = inference[:-2]
    else :
        inference = inference.rstrip(').')
    return clause_id, role, formula, inference

### CALL AGINT FOR RATING ###

def enrich_graph_with_agint(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Enriches the graph with AGInT scores in a compact manner.
    
    Args:
        graph: NetworkX graph with DerivationNode objects as node data.
        
    Returns:
        The enriched graph with AGInT scores.
    """
    
    # 1. Extract all CNF clauses
    tptp_content = _extract_tptp_from_graph(graph)
    
    # 2. Call the AGInT API
    agint_output = _call_agint(tptp_content)
    if not agint_output:
        return graph
    # 3. Parse the output and update the nodes
    scores_map = _parse_agint_scores(agint_output)
    
    for node_id in graph.nodes():
        node = graph.nodes[node_id]['data']
        if node.clause_id in scores_map:
            score_data = scores_map[node.clause_id]
            node.interesting_score = score_data.get('interesting', 0.0)
            node.other_agint_metrics = {k: v for k, v in score_data.items() 
                                       if k != 'interesting'}
    
    return graph


def _extract_tptp_from_graph(graph: nx.DiGraph) -> str:
    """Extracts all CNF clauses from the graph in a TPTP format suitable for AGInT."""
    
    clauses = []
    for node_id in graph.nodes():
        node = graph.nodes[node_id]['data']
        
        # Take the formula as is
        formula = node.clause_formula
        
        # Simple source: use clause_id if no inference, otherwise clean up the inference string
        if not node.inference or node.inference.startswith('file('):
            clause = f"cnf({node.clause_id},{node.role},{formula})."
            clauses.append(clause)
        else:
            source = node.inference
            clause = f"cnf({node.clause_id},{node.role},{formula},{source})."
            clauses.append(clause)
        
    return "\n".join(clauses)


def _call_agint(tptp_content: str) -> str:
    """Calls the AGInT with udocker and returns the output."""
    return prover_session.run_agint(tptp_content)
            


def _parse_agint_scores(agint_output: str) -> dict:
    """Parses the AGInT output and returns a dictionary {clause_id: {metric: score}}."""
    
    scores_map = {}
    pattern = r'cnf\(([^,]+),.*?(\[.*?\])\s*\)\.'
    
    for match in re.finditer(pattern, agint_output, re.DOTALL):
        clause_id = match.group(1).strip()
        scores_str = match.group(2) 
        
        scores = {}
        for score_match in re.finditer(r'(\w+)\(([^)]+)\)', scores_str):
            metric = score_match.group(1).lower()
            value_str = score_match.group(2).strip()
            
            if value_str.lower() != "ignored":
                try:
                    scores[metric] = float(value_str)
                except ValueError:
                    pass
        
        if scores:
            scores_map[clause_id] = scores
            
    return scores_map

