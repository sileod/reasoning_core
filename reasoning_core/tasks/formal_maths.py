import networkx as nx
import re
import os
import tempfile
import random
import json
import gzip
from easydict import EasyDict as edict
from dataclasses import dataclass
from appdirs import AppDirs
from pathlib import Path
from reasoning_core.utils.udocker_process import prover_session
from ._sat_graph import generate_derivation_graph
from reasoning_core.template import Task, Problem, Config
import ast

def extract_problem_from_graph(G: nx.DiGraph, node_id_str: str, max_length_proof: int):
    theorem = G.nodes[node_id_str]['data'].clause_formula
    frontier = {node_id_str}
    collected_hypotheses = set()
    
    for _ in range(max_length_proof):
        nxt = set()
        for v in frontier:
            parents = list(G.predecessors(v))
            if parents:
                # Continue traversing up the graph
                nxt.update(parents)
            else:
                # FIX: Capture leaves (axioms) encountered on short branches
                collected_hypotheses.add(v)
        
        if not nxt:
            break
        frontier = nxt
    
    # The final frontier (nodes at max_depth) are also hypotheses
    collected_hypotheses.update(frontier)
    
    hypotheses = [G.nodes[n]['data'].clause_formula for n in collected_hypotheses]
    hypotheses = [h for h in hypotheses if h != theorem]
    return hypotheses, theorem

def extract_useful_axioms(G: nx.DiGraph, node_id_str: str) : 
    ancestors = nx.ancestors(G, node_id_str)

    initial_ax = {n for n, in_degree in G.in_degree() if in_degree == 0}

    useful_ax = ancestors.intersection(initial_ax)

    return useful_ax


def normalize_formula(f: str) -> str:
    """Canonicalize formula: remove whitespace and anonymize variables."""
    if not f: return ""
    # Remove whitespace
    f = re.sub(r"\s+", "", f)
    # Replace variables (e.g., X123) with generic V to handle alpha-equivalence
    f = re.sub(r"X\d+", "V", f)
    return f

# 2. FIX: Clean CoT generation with step collapsing and better labeling
def make_cot(G: nx.DiGraph, target_node: str, formula_map: dict) -> str:
    sub = G.subgraph(nx.ancestors(G, target_node) | {target_node})
    lines = []
    node_to_label = {}
    step_counter = 0
    sys_ax_counter = 0  # Fixed variable name

    # Topological sort ensures we process parents before children
    for node in nx.topological_sort(sub):
        
        # Optimization: Skip intermediate 1-parent nodes (normalization/copy steps)
        # This removes "c_0_X" noise unless it's the final theorem
        parents = sorted(list(sub.predecessors(node)))
        if len(parents) == 1 and node != target_node:
            # Inherit label from the single parent (collapse step)
            p_lbl = node_to_label.get(parents[0])
            if p_lbl:
                node_to_label[node] = p_lbl
                continue

        data = sub.nodes[node]['data']
        f_norm = normalize_formula(data.clause_formula)
        
        val = formula_map.get(f_norm)
        is_theorem = (node == target_node)
        
        # Determine Label
        if not parents:
            # Leaf / Axiom
            if is_theorem:
                label = "THEOREM"
                lines.append(f"THEOREM [ '{data.clause_formula.strip()}' ] (axiom)")
            elif val is not None and str(val) != "THEOREM":
                label = f"premise_{val}"
            else:
                # Fallback for unmapped system axioms
                label = f"sys_ax_{sys_ax_counter}"
                sys_ax_counter += 1
            node_to_label[node] = label
            continue

        # Derived Node
        # Get parent labels
        p_labels = [node_to_label.get(p) for p in parents if p in node_to_label]
        if not p_labels: continue

        if is_theorem:
            label = "THEOREM"
        else:
            label = f"step_{step_counter}"
            step_counter += 1
        
        node_to_label[node] = label

        # Clean Inference Rule Name
        # Extract 'res', 'pm', 'rw' from string like "inference(rw,[status...])"
        inf_str = data.inference or ""
        rule_match = re.search(r'inference\(([a-zA-Z0-9_]+)', inf_str)
        if rule_match:
            rule_name = rule_match.group(1)
        else:
            # Fallback cleanup for non-standard formats
            rule_name = re.match(r'([a-zA-Z0-9_]+)', inf_str).group(1) if inf_str else "inference"
            if rule_name.startswith("c_0"): rule_name = "processing"

        lines.append(f"{label} {rule_name}({', '.join(p_labels)}): [ '{data.clause_formula.strip()}' ]")

    return "\n".join(lines).strip()

def perturb_list(input_l: list, base_domain: list, n_perturbations: int = 1) -> list:
    """Applies cumulative perturbations to a list."""
    lst = list(input_l) 
    base_set = set(base_domain)

    for _ in range(n_perturbations):
        complementary = base_set - set(lst)
        
        possible_ops = []
        if complementary:
            possible_ops.append('add')
            if lst: 
                possible_ops.append('replace')
        if len(lst) > 1:
            possible_ops.append('remove')
        if not possible_ops:
            break
            
        op_type = random.choice(possible_ops)
        
        if op_type == 'add':
            lst.insert(random.randint(0, len(lst)), random.choice(list(complementary)))
        elif op_type == 'remove':
            lst.pop(random.randint(0, len(lst) - 1))
        elif op_type == 'replace':
            index_to_replace = random.randint(0, len(lst) - 1)
            lst[index_to_replace] = random.choice(list(complementary))
            
    return lst

def prove_conjecture(axioms: list[str], conjecture: str,
                        time_limit_seconds: str ="2d", verb: bool = False):
    """
    Uses Vampire to prove or disprove a conjecture given a set of axioms.
    Returns True (provable), False (disprovable/countersatisfiable), or an error string.
    """
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.p') as temp_f:
        for i, axiom in enumerate(axioms, 1):
            temp_f.write(f"cnf(axiom_{i}, axiom, {axiom}).\n")
        temp_f.write(f"fof(conjecture_1, conjecture, {conjecture}).\n")
        temp_f.flush()
        
        if verb == True:
            print(f"---- proof file :-------------------------")
            temp_f.seek(0)  
            print(temp_f.read()) 
            print("-------------------------------------------------")


        vampire_command_proove = [ "-t", str(time_limit_seconds)]

        vampire_command_disproove = ["-t", str(time_limit_seconds),"-sa", "fmb"]

        result_proove = prover_session.run_prover('vampire',vampire_command_proove,temp_f.name)

        if verb == True:
            print(f"output proove vampire :  {result_proove.stdout} ")

        if "% SZS status Theorem" in result_proove.stdout :
            return True
        if "% SZS status CounterSatisfiable" in result_proove.stdout :
            return False

        result_disproove = prover_session.run_prover('vampire',vampire_command_disproove,temp_f.name)
    
        if verb == True:
            print(f"output disproove vampire :  {result_disproove.stdout} ")

        if "% Finite Model Found!" in result_disproove.stdout :
            return False 
        if "% Time limit reached!" in result_proove.stdout and "% Time limit reached!" in result_disproove.stdout  :
            return f"ERROR : TIME LIMIT in both tentative to proove AND to disproove"
        else :
            return f"ERROR : {result_proove.stderr}{result_disproove.stderr}"
        

dirs = AppDirs("Axioms_TPTP")
BASE_DIR = Path(__file__).resolve().parent.parent
AXIOM_ARCHIVE_PATH = BASE_DIR / "resources" / "axioms_filtered.json.gz"
DOMAIN_MAP = {
    'ALG': 'Algebra',
    'ANA': 'Analysis',
    'FLD': 'Field Theory',
    'GEO': 'Geometry',
    'GRP': 'Group Theory',
    'LCL': 'Logic Calculi',
    'NUM': 'Number Theory',
    'RNG': 'Ring Theory',
    'SET': 'Set Theory',
    'TOP': 'Topology'
}

def get_random_tptp_axioms(
    axiom_archive=AXIOM_ARCHIVE_PATH, 
    prefixes=None, 
    cache_dir=dirs.user_cache_dir ):
    
    try:
        with gzip.open(axiom_archive, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, EOFError):
        return None, None

    keys = list(data.keys())
    if prefixes:
        keys = [k for k in keys if k.startswith(tuple(prefixes))]

    if not keys:
        return None, None
        
    chosen_key = random.choice(keys)
    content = data[chosen_key]

    os.makedirs(cache_dir, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        encoding='utf-8', 
        suffix='.p', 
        dir=cache_dir,
        delete=False  
    )
    
    with temp_file:
        temp_file.write(content)
        temp_file.flush()

    return temp_file.name, chosen_key

@dataclass
class EntailConfig(Config):
    proof_depth: int = 1
    perturbation: int = 1
    min_interesting_score: float = 0.6
    positive_problem_ratio: float = 0.25
    domains = ['ALG', 'ANA', 'FLD', 'GEO', 'GRP', 'LCL', 'NUM', 'RNG', 'SET', 'TOP']

    def update(self, c):
        self.proof_depth += c
        self.perturbation += c

class ConjectureEntailment(Task):
    """
    A task that generates problems to determine if a set of hypotheses
    proves a given conjecture.
    """
    def __init__(self, config=EntailConfig()):
        super().__init__(config)

    def _initialize_graph(self):    
        for _ in range(100):
            axiom_file_path, axiom_file_name = get_random_tptp_axioms(prefixes=self.config.domains)

            if axiom_file_path:
                self.axiom_set = axiom_file_name
            self.graph = generate_derivation_graph( 
                    axiom_file = axiom_file_path, 
                    save_output=False, 
                    ranking=True, 
                    e_limit=2
                )
            if os.path.exists(axiom_file_path):
                os.remove(axiom_file_path)
            

            self.all_formulas = [data['data'].clause_formula for _, data in self.graph.nodes(data=True)]
            self.interesting_thm = []

            for i in self.graph.nodes() : 
                if self.graph.nodes[i]['data'].interesting_score > self.config.min_interesting_score and self.graph.in_degree(i) > 1 :
                    self.interesting_thm.append(i)
            if len(self.interesting_thm) >= 5 :
                break

    def generate(self):
        self._initialize_graph()

        while True :
            
            theorem_node_id = random.choice(list(self.interesting_thm))
            correct_hypotheses, theorem = extract_problem_from_graph(self.graph, theorem_node_id, self.config.proof_depth)
            useful_axioms = extract_useful_axioms(self.graph, theorem_node_id)
            useful_axioms_formula = [self.graph.nodes[node]['data'].full_cnf_clause for node in useful_axioms]
            if random.random() < self.config.positive_problem_ratio:
                hypotheses = correct_hypotheses
                if prove_conjecture(hypotheses, theorem) is not True:
                    continue
                answer = True 
            else:
                distraction_pool = list(set(self.all_formulas) - {theorem})
                hypotheses = perturb_list(correct_hypotheses, distraction_pool ,self.config.perturbation)
                try:
                    answer = prove_conjecture(hypotheses, theorem)
                except TimeoutError:
                    continue

            if isinstance(answer, bool):
                metadata = edict({'hypotheses': hypotheses,
                            'conjecture': theorem,
                            'correct_hypotheses': correct_hypotheses ,
                            'proof_depth' : self.config.proof_depth,
                            'perturbation' : self.config.perturbation ,
                            'useful_axioms' : useful_axioms_formula,
                            'axiom_set' : self.axiom_set})
                return Problem(metadata, str(answer))

    def prompt(self, metadata):

        axiom_text = "\n".join([f"- {h}" for h in metadata['useful_axioms']])
        hypotheses_text = "\n".join([f"- {h}" for h in metadata['hypotheses']])
        domain_name = DOMAIN_MAP.get(metadata['axiom_set'][:3], metadata['axiom_set'])

        return (
            f"You are a mathematical logic assistant. Your task is to determine the sufficiency of a specific set of axioms for proving a theorem.\n\n"
            f"By using the **Superposition Calculus** (which includes rules like Resolution and Paramodulation).\n"
            f"## General Context\n"
            f"The problem is set in the domain of: **{domain_name}**.\n"
            f"The following are the fundamental axioms of this domain, providing a general theoretical background:\n"
            f"Fundamental Axioms:\n"
            f"{axiom_text}\n\n"
            f"## Task\n"
            f"Now, you are given a specific **subset of axioms** and a theorem from this domain.\n\n"
            f"**Axiom Subset under consideration:**\n"
            f"{hypotheses_text}\n\n"
            f"**Theorem to prove:**\n"
            f"`{metadata['conjecture']}`\n\n"
            f"### Question\n"
            f"Is the **\"Axiom Subset under consideration\"** listed above **sufficient on its own** to prove the **\"Theorem to prove\"**?\n\n"
            f"### Response Format\n"
            f"Respond only with `True` if the provided subset is sufficient, or `False` otherwise. Do not provide explanations."
        )

    def score_answer(self, answer, entry):
        ref = entry.answer.lower()
        pred = str(answer).lower().strip().strip('"').strip("'")
        return float(ref==pred)


@dataclass
class SelectionConfig(Config):
    proof_depth: int = 1
    min_interesting_score: float = 0.6
    num_distractors: int = 2
    domains = ['ALG', 'ANA', 'FLD', 'GEO', 'GRP', 'LCL', 'NUM', 'RNG', 'SET', 'TOP']

    def update(self, c):
        self.proof_depth += c
        self.num_distractors += c


class TheoremPremiseSelection(Task):
    """
    A task that generates problems where one must select the essential hypotheses
    required to prove a given conjecture from a larger pool of axioms.
    And a minimality check to ensure the ground truth is correct.
    """
    def __init__(self, config=SelectionConfig()):
        super().__init__(config)
        
    _initialize_graph = ConjectureEntailment._initialize_graph

    def _reprove_with_minimal(self, hypotheses: list) -> nx.DiGraph:
            """
            Run E-prover on ONLY the minimal set as AXIOMS. 
            No conjecture is passed; we rely on derivation to find the theorem node.
            """
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.p', delete=True) as tf:
                for i, h in enumerate(hypotheses):
                    tf.write(f"cnf(h_{i}, axiom, {h}).\n")
                tf.flush()
                # rank=False prevents unnecessary processing
                return generate_derivation_graph(tf.name, save_output=False, e_limit=2, ranking=False)

    def find_minimal_hypotheses(self, initial_hypotheses: list[str], conjecture: str) -> list[str]:
        """
        Prunes an initial set of hypotheses down to a minimal subset that is
        still sufficient to prove the conjecture.
        """
        essential_hypotheses = set(initial_hypotheses)
        
        for h in initial_hypotheses:
            
            temp_set = essential_hypotheses.copy()
            if h in temp_set:
                temp_set.remove(h)
            else:
                continue 

            is_provable = prove_conjecture(list(temp_set), conjecture)
            
            if is_provable is True:
                essential_hypotheses.remove(h)
                
        return list(essential_hypotheses)

    def generate(self):
        self._initialize_graph()
    
        while True:
            if not self.interesting_thm:
                self._initialize_graph()
                if not self.interesting_thm: continue

            theorem_node_id = random.choice(self.interesting_thm)
            
            # 1. Extract Superset & Minimize
            superset, theorem = extract_problem_from_graph(
                self.graph, theorem_node_id, self.config.proof_depth
            )
            
            try:
                # Verify superset (optimization)
                if prove_conjecture(superset, theorem) is not True: continue
                
                minimal = self.find_minimal_hypotheses(superset, theorem)
                
                # Verify minimal (safety)
                if not minimal or prove_conjecture(minimal, theorem) is not True: continue
            except Exception: continue

            # 2. RE-PROVE for Clean CoT (Forward Derivation)
            clean_graph = self._reprove_with_minimal(minimal)
            
            # Locate theorem node in new graph
            target_node = None
            clean_theorem_str = normalize_formula(theorem)
            
            for n, d in clean_graph.nodes(data=True):
                if normalize_formula(d['data'].clause_formula) == clean_theorem_str:
                    target_node = n
                    break
            
            if not target_node: continue 

            # 3. Create Distractors & Pool
            distractor_pool = list(set(self.all_formulas) - set(minimal) - {theorem})
            if len(distractor_pool) < self.config.num_distractors: continue 
            
            distractors = random.sample(distractor_pool, self.config.num_distractors)
            pool = minimal + distractors
            random.shuffle(pool)

            # 4. Generate CoT
            # Map ONLY minimal premises to their pool indices.
            # This ensures distractors (if derived coincidentally) aren't labeled as premises.
            f_map = {normalize_formula(h): pool.index(h)+1 for h in minimal}
            f_map[clean_theorem_str] = "THEOREM"

            cot = make_cot(clean_graph, target_node, f_map)

            # 5. Metadata & Context Filtering
            pool_norm = set(normalize_formula(h) for h in pool)
            useful_axioms_norm = []
            orig_useful_ids = extract_useful_axioms(self.graph, theorem_node_id)
            
            for uid in orig_useful_ids:
                u_cnf = self.graph.nodes[uid]['data'].full_cnf_clause
                if normalize_formula(self.graph.nodes[uid]['data'].clause_formula) not in pool_norm:
                    useful_axioms_norm.append(u_cnf)

            metadata = edict({
                'hypotheses_pool': pool, 
                'theorem': theorem,
                'cot': cot,
                'correct_indices': sorted([pool.index(h) + 1 for h in minimal]),
                'correct_minimal_hypotheses': minimal, 
                'useful_axioms': useful_axioms_norm, 
                'axiom_set': self.axiom_set
            })
            
            return Problem(metadata, str(metadata.correct_indices))

    def prompt(self, metadata):
    
        axiom_text = "\n".join([f"- {h}" for h in metadata['useful_axioms']])
        hypotheses_text = "\n".join(
            [f"{i+1}. {h}" for i, h in enumerate(metadata['hypotheses_pool'])]
        )
        domain_name = DOMAIN_MAP.get(metadata['axiom_set'][:3],metadata['axiom_set'])

        
        return (
            f"You are a mathematical logic assistant. Your task is to identify a minimal set of premises sufficient for a proof.\n\n"
            f"By using the **Superposition Calculus** (which includes rules like Resolution and Paramodulation).\n"
            f"## General Context\n"
            f"The problem is set in the domain of: **{domain_name}**.\n"
            f"The following are the fundamental axioms of this domain. They provide general context. **Do not use them in the proof itself.**\n"
            f"Fundamental Axioms:\n"
            f"{axiom_text}\n\n"
            f"## Task\n"
            f"Your goal is to prove the following theorem:\n"
            f"**Theorem:**\n"
            f"`{metadata['theorem']}`\n\n"
            f"Below is a numbered pool of potential premises. Your task is to identify the **minimal subset** of numbers from this pool whose corresponding statements are **sufficient on their own** to prove the theorem.\n"
            f"**Pool of Premises:**\n"
            f"{hypotheses_text}\n\n"
            f"### Question\n"
            f"Which is the smallest set of numbered premises from the pool that is sufficient to prove the theorem, without using the fundamental axioms from the context?\n\n"
            f"### Response Format\n"
            f"Your answer must be **only** a list of numbers, sorted in increasing order. For example: `[2, 5, 8]`."
        )


    def score_answer(self, answer, entry):
        """
        Scores the answer using the Jaccard Index .
        """
        metadata = entry.metadata
        hypotheses_pool = metadata.get('hypotheses_pool')
        if not hypotheses_pool:
            return 0.0


        truth_indices = set(ast.literal_eval(entry.answer))
        pred_indices = set(map(int, re.findall(r'\d+', str(answer))))


        intersection = len(truth_indices.intersection(pred_indices))
        union = len(truth_indices.union(pred_indices))

        if union == 0:
            return 1.0  

        return intersection / union


@dataclass
class ReconstructionConfig(Config):
    proof_depth: int = 1
    min_interesting_score: float = 0
    domains = ['ALG', 'ANA', 'FLD', 'GEO', 'GRP', 'LCL', 'NUM', 'RNG', 'SET', 'TOP']

    def update(self, c):
        self.proof_depth += c

class ProofReconstruction(Task):
    """
    A task that generates problems where one must reconstruct the derivation
    graph from a numbered list of shuffled clauses.
    """
    def __init__(self, config=ReconstructionConfig()):
        super().__init__(config)
        
    _initialize_graph = ConjectureEntailment._initialize_graph
    

    def generate(self):

        self._initialize_graph()
        useless_axioms = {n for n, d in self.graph.in_degree() if d == 0}

        redundant_children = set()
        for ax_id in useless_axioms:
            if self.graph.out_degree(ax_id) == 1:
                child_id = list(self.graph.successors(ax_id))[0]
                if self.graph.nodes[ax_id]['data'].clause_formula == self.graph.nodes[child_id]['data'].clause_formula:
                    redundant_children.add(child_id)
        nodes_to_remove = useless_axioms.union(redundant_children)

        self.graph.remove_nodes_from(nodes_to_remove)
            
        all_axioms = {node for node, in_degree in self.graph.in_degree() if in_degree == 0}
        
        interesting_theorems = self.interesting_thm

        valid_paths = []
        for theorem_id in interesting_theorems:
            ancestor_axioms = nx.ancestors(self.graph, theorem_id) & all_axioms
            
            for axiom_id in ancestor_axioms:
                path_length = nx.shortest_path_length(self.graph, source=axiom_id, target=theorem_id)
                
                if 0 < path_length <= self.config.proof_depth:
                    
                    proof_nodes = nx.ancestors(self.graph, theorem_id)
                    proof_nodes.add(theorem_id)
                    num_nodes = len(proof_nodes)
                    min_size = 2**(self.config.proof_depth) - 1
                    max_size = 2**(self.config.proof_depth+1) - 1
                    
                    if min_size < num_nodes <= max_size:

                        is_binary = all(
                            self.graph.in_degree(n) in (0, 2) for n in proof_nodes
                        )

                        if is_binary:
                            valid_paths.append((axiom_id, theorem_id))
                            break 

        if not valid_paths:
            return None

        axiom_id, theorem_node_id = random.choice(valid_paths)
        
        proof_nodes = nx.ancestors(self.graph, theorem_node_id)
        proof_nodes.add(theorem_node_id)
        proof_graph = self.graph.subgraph(proof_nodes)

        all_clauses_in_proof = [data['data'].clause_formula for _, data in proof_graph.nodes(data=True)]
        random.shuffle(all_clauses_in_proof)
        theorem_formula = self.graph.nodes[theorem_node_id]['data'].clause_formula

        proof_structure_indices = []

        for node_id in proof_graph.nodes():
            parents = list(proof_graph.predecessors(node_id))
            if parents:  
                child_formula = proof_graph.nodes[node_id]['data'].clause_formula
                parent_formulas = [proof_graph.nodes(data=True)[p]['data'].clause_formula for p in parents]
                
                child_idx = all_clauses_in_proof.index(child_formula) + 1
                parent_indices = sorted([all_clauses_in_proof.index(p) + 1 for p in parent_formulas])
    
                proof_structure_indices.append(f"{child_idx} <- {', '.join(map(str, parent_indices))}")

        proof_structure_ids = [f"{node} <- {', '.join(sorted(list(proof_graph.predecessors(node))))}" for node in proof_graph.nodes() if proof_graph.in_degree(node) > 0]
        

        f_map = {normalize_formula(c): i+1 for i, c in enumerate(all_clauses_in_proof)}
        cot = make_cot(proof_graph, theorem_node_id, f_map)

        metadata = edict({
            'numbered_clauses': all_clauses_in_proof, 
            'conjecture': theorem_formula,
            'cot': cot,
            'correct_proof_structure_indices' : proof_structure_indices,
            'correct_proof_structure_ids': sorted(proof_structure_ids),
            'correct_proof_graph' : str(proof_graph),
            'proof_depth' : self.config.proof_depth,
            'axiom_set': self.axiom_set
        })

        answer = '\n'.join(str(element) for element in sorted(proof_structure_indices))
        return Problem(metadata, answer)

    def prompt(self, metadata):
        
        clauses_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(metadata['numbered_clauses'])])
        domain_name = DOMAIN_MAP.get(metadata['axiom_set'][:3], metadata['axiom_set'])


        return (
            f"Your task is to reconstruct the dependency graph of a mathematical proof from the domain of **{domain_name}**.\n\n"
            f"The proof graph concludes with the theorem: `{metadata['conjecture']}`\n\n"
            f"## Proof Context & Rules\n"
            f"This proof was generated by using the **Superposition Calculus** (which includes rules like Resolution and Paramodulation).\n\n"
            f"Therefore, the proof has the following properties:\n"
            f"- **Starting Points:** Some clauses in the list are starting points (axioms ) and are not derived from other clauses.\n"
            f"- **Derived Clauses:** Every other clause is derived from exactly **two** parent clauses from the list.\n"
            f"- **Clause Reuse:** A single clause can be used as a parent in multiple derivation steps.\n\n"
            f"## Your Task\n"
            f"Given the rules above, reconstruct the proof from the following shuffled list of clauses. Identify the derivation for every clause that is not a starting point.\n\n"
            f"**Shuffled Clauses:**\n"
            f"{clauses_text}\n\n"
            f"## Required Output Format\n"
            f"- List **only** the derivation steps.\n"
            f"- Each step must be on a new line.\n"
            f"- Use the exact format `CHILD <- PARENT_1, PARENT_2`. Example: `5 <- 2, 4`.(for each line)\n"
            f"- All clauses from the list must be used in the final structure.\n"
            f"- No explanations, comments, or extra text."
        )
        
    def score_answer(self, answer, entry):
        """
        Steps:
        1) Structural coherence: strict parsing + DAG + 2 distinct parents per child + unique child.
        2) "Gold overlap" score: 1.0 for exact match; otherwise F1 between prediction and reference.
        3) Semantic score: ratio of (sampled) steps validated by prove_conjecture.
        4) Combination: score = w_gold * F1 + (1 - w_gold) * semantic_ratio.
        """

        import re, random
        try:
            import networkx as nx
        except Exception:
            # If networkx is unavailable, DAG cannot be verified
            return 0.0

        clauses_pool = entry.metadata.get('numbered_clauses')
        gold = entry.metadata.get('correct_proof_structure_indices') or []
        if not clauses_pool:
            return 0.0

        n = len(clauses_pool)

        # Hyperparameters
        w_gold = 0.7  # weight of the overlap vs. the semantic verification
        max_semantic_checks = 50  # limit of prover calls to keep it fast

        # 1) Strict parsing
        lines = [l.strip() for l in str(answer).splitlines() if l and l.strip()]
        if not lines:
            return 0.0

        pat = re.compile(r'^\s*(\d+)\s*<-\s*(\d+)\s*,\s*(\d+)\s*$')
        derivations = []
        seen_children = set()
        used_indices = set()

        for line in lines:
            m = pat.fullmatch(line)
            if not m:
                return 0.0  # non-compliant format

            child, p1, p2 = map(int, m.groups())

            # index bounds
            if not (1 <= child <= n and 1 <= p1 <= n and 1 <= p2 <= n):
                return 0.0

            # 2 distinct parents, different from the child
            if p1 == p2 or child in (p1, p2):
                return 0.0

            # each child must be defined only once
            if child in seen_children:
                return 0.0
            seen_children.add(child)

            p1, p2 = sorted((p1, p2))
            derivations.append((child, p1, p2))
            used_indices.update((child, p1, p2))

        # 2) Graph and minimal coherence (DAG)
        g = nx.DiGraph()
        g.add_nodes_from(range(1, n + 1))
        for child, p1, p2 in derivations:
            g.add_edge(p1, child)
            g.add_edge(p2, child)

        if not nx.is_directed_acyclic_graph(g):
            return 0.0

        # exact in-degree for derived nodes (robustness)
        for child in seen_children:
            if g.in_degree(child) != 2:
                return 0.0

        # 3) Comparison to ground truth (exact-match then F1)
        pred_set = set(f"{c} <- {p1}, {p2}" for (c, p1, p2) in derivations)
        gold_set = set(gold)

        # Exact match (regardless of order) -> maximum score
        if gold_set and pred_set == gold_set:
            return 1.0

        # Partial F1 if a ground truth is available
        f1 = 0.0
        if gold_set:
            tp = len(pred_set & gold_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # 4) Local semantic verification via the prover (sampled)
        sem_ratio = 0.0
        to_check = derivations
        if len(to_check) > max_semantic_checks:
            to_check = random.sample(to_check, max_semantic_checks)

        if to_check:
            succ = 0
            total = 0
            for child, p1, p2 in to_check:
                try:
                    axioms_list = [clauses_pool[p1 - 1], clauses_pool[p2 - 1]]
                    conj = clauses_pool[child - 1]
                    ok = None
                    # Main attempt: usual named parameters
                    try:
                        ok = prove_conjecture(axioms=axioms_list, conjecture=conj)
                    except TypeError:
                        # Fallbacks (depending on existing signature)
                        try:
                            ok = prove_conjecture(axioms=axioms_list, conjecture=conj)
                        except Exception:
                            ok = prove_conjecture(axioms_list, conj)  # positional
                    if ok is True:
                        succ += 1
                    total += 1
                except Exception:
                    # we ignore errors from an individual step
                    total += 1
            if total:
                sem_ratio = succ / total

        # 5) Weighted combination (and clamp)
        if gold_set:
            score = w_gold * f1 + (1.0 - w_gold) * sem_ratio
        else:
            score = sem_ratio  # if there's no gold, we can only score semantically

        return float(max(0.0, min(1.0, score)))
