import random
import sympy as sp
from unigram import init_grammar, generate
import re
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy as dc


from reasoning_core.template import Task, Problem, Config


# ----cfg for formulas---- 📋

def Sequence_cfg(mode: str, recurrence_depth: int): #chatgpt one
    """ Generate a grammar of recursive sequences of a given depth.

    Args:
        mode ("simple" or "full"): type of sequence
        recurrence_depth (int): the maximum recursion depth

    Returns:
        Rule subclass: a unigram grammar object for recursive sequences
    """
    R = init_grammar(['eq'], name=f"sequence_of_depth_{recurrence_depth}", preprocess_template=lambda s:s)

    # --- Common rules ---
    R('start(exp)', '{0}')            # top-level expansion
    R('exp', 'n')                     # U[n] = n

    # U[n] = U[n - d]
    R('exp(Ui)', '{0}')
    for i in range(1, recurrence_depth + 1):
        R('Ui', f'U{i}')

    # Constant terms
    R('exp(c_z)', '{0}')
    for i in range(-9, 10):
        R('c_z', str(i))

    # Binary operations
    R('exp(exp,bop,exp)', '({0} {1} {2})')
    for op in ['+', '*', '-']:
        R('bop', op)

    # --- Full-mode extensions ---
    if mode == 'full':
        # Unary operations like sign() or relu()
        R('exp(uop,exp)', '{0}({1})')
        for u in ['sign', 'relu']:
            R('uop', u)

        # Safe binary ops with natural-number constants
        R('exp(exp,safe_bop,c_n)', '({0} {1} {2})')
        for i in range(1, 10):
            R('c_n', str(i))
        for op in ['/', '%']:
            R('safe_bop', op)

    return R

# ---- The underlying object class (Sequence) ----  📟

class Sequence:
    def __init__(self, formula : str, initial_elem : list = None):
        self.rec_formula = formula
        self.degree = self.compute_degree()
        if initial_elem == None:
            self.set_random_first_sample()
        elif (self.degree <= len(initial_elem)):
            self.first_elem = initial_elem[:self.degree]
        else:
            raise ValueError(f"The degree of the recursive formula is {self.degree}, but there is only {len(initial_elem)} initial terms, so the sequence cannot be properly defined.")

    def __repr__(self):
        return f"The formula of the sequence is {self.rec_formula}, of degree of recursion {self.degree}, and initial terms {self.first_elem}"

    def compute_degree(self) -> int: 
        ret = 0
        for i in range(100):
            U_i = f'U[n - {i}]'
            U_i_bis = f'U[n-{i}]' #case of LLM not respecting format
            if U_i in str(self.rec_formula) or U_i_bis in str(self.rec_formula):
                ret = i
        return ret

    def instantiate(self, previous_terms : list, rank : int) -> str:
        """instantiate the U[n - i] and n within the recursive formula of the Sequence, used in the function U_n in order to compute the following element"""
        d = self.degree
        if d > len(previous_terms) or d > rank:
            raise ValueError(f"""Cannot instantiate, as the recursive degree of the formula ( {d} )
             is superior to the rank ( {rank} ) or to the number of the previous provided terms ( {len(previous_terms)} )""")
        ret = str(self.rec_formula)
        for i,value in enumerate(previous_terms[-d:]): # replace each U[n - i] by its value
            index = d-i
            to_replace = f'U[n - {index}]'
            to_replace_bis = f'U[n-{index}]' # case of llm not respecting format
            ret = ret.replace(to_replace, str(value))
            ret = ret.replace(to_replace_bis, str(value))
        ret = re.sub(r'(?<!sign)\bn\b', str(rank), ret) # replace "n" by the rank, except the case when "n" is in "sign"
        return ret

    def set_random_first_sample(self):
        self.first_elem = [np.random.randint(-9,10) for _ in range(self.degree)]

    def U_n(self, predecessors : list, rank : int) -> int :
        """Given the rank and the predecessors, compute the following element of the sequence"""
        formula = str(self.rec_formula)
        d = self.degree
        formula_instance = self.instantiate(predecessors, rank = rank)
        relu = lambda x : max(0,x)  # define those functions for the eval
        Mod = lambda k,n : k%n 
        sign = lambda n : 1 if n > 0 else -1 if n < 0 else 0
        return eval(formula_instance.replace('/', '//')) # in order to use integer division

    def n_first_elem(self, n: int, max_terms_len: int = 12) -> list:
        """
        Generates the first n elements of the sequence.

        Args:
            n (int): The number of elements to generate.
            max_terms_len (int, optional): The maximum string length allowed for any single term.
                                    If a term exceeds this length, the generation stops,
                                    and the list of terms generated so far is returned.

        Returns:
            list: A list of the sequence elements. The list will have a length of 'n'
                if the generation completed, or a length less than 'n' if it was
                stopped early by the max_terms_len check.
        """
        if n < self.degree:
            raise ValueError("The number of sequence elements requested is less than its degree of recurrence.")
        
        ret = self.first_elem[:]
        rank = len(ret)
        while len(ret) < n:
            next_term = self.U_n(predecessors=ret, rank=rank)
            if max_terms_len is not None and len(str(next_term)) > max_terms_len:
                # Explosion detected! Stop the process gracefully.
                break
            ret.append(next_term)
            rank += 1
            
        return ret

# --- Filters for generations --- ✂️

def filter_2_outof(S :Sequence, length_check : int = 10) -> bool:
    """Filter constant sequences"""
    first_terms = S.n_first_elem(length_check)[S.degree:]
    return len(set(first_terms)) > 1


def filter_max_terms_len(S :Sequence, max_terms_len : int = 12, length_check : int = 12) -> bool:
    """Check if, for the length_check first terms, the length of the numbers of the sequence doesn't exceed max_terms_len (prevent explosions)"""
    first_terms = S.n_first_elem(length_check, max_terms_len= max_terms_len) # might not compute a safe version of "n_first_elem"
    return max([ len(str(elem)) for elem in first_terms ]) <= max_terms_len and len(first_terms) == length_check

# --- Formula generator class  --- 🏡


@dataclass
class SequenceConfig(Config):
    mode= "simple" #can be 'full' as well
    recurrence_depth: int = 1
    n_visible_terms: int = 8
    max_terms_len: int = 15
    min_depth_grammar: int = 1.5
    max_depth_grammar: int = 3
    def update(self, c):
        self.recurrence_depth += c
        self.n_visible_terms += 2 * c
        self.min_depth_grammar += 0.5 * c
        self.max_depth_grammar += c


class SequentialInduction(Task):
    def __init__(self, config=SequenceConfig()):
        super().__init__(config=config)
        # Now, self.filters will contain references to picklable instance methods
        self.filters = [
            self._filter_2_outof,
            self._filter_max_terms_len
        ]

    def _filter_2_outof(self, s):
        return filter_2_outof(s, length_check=self.config.n_visible_terms)

    def _filter_max_terms_len(self, s):
        return filter_max_terms_len(
            s,
            max_terms_len=self.config.max_terms_len,
            length_check=self.config.n_visible_terms + 1
        )
    
    def one_shot_sympy_generate(self):
        """generate a formula (instance of the defined cfg) and simplify it with sympy"""
        rule = Sequence_cfg(self.config.mode, self.config.recurrence_depth)
        prod = generate(rule, depth=self.config.max_depth_grammar, min_depth=self.config.min_depth_grammar)@'eq' 
        return convert_to_sympy(prod, self.config.recurrence_depth)

    
    def generate(self) -> Problem:
        formula = self.one_shot_sympy_generate()
        S = Sequence(formula)
        while not all([ self.filters[i](S) for i in range(len(self.filters)) ]): #keep generating until the sequence fulfill all requirements (filters)
            formula = self.one_shot_sympy_generate()
            S = Sequence(formula)
        data = {"first elements" : S.n_first_elem(self.config.n_visible_terms), "degree of recursion" : S.degree, "initial terms" : S.first_elem}
        answer = str(formula)
        return Problem(metadata = data, answer = answer)

    def verify(self, y_pred, y_truth, initial_element = None) -> bool:
        """ Check if the guessed formula match with the true one (for the n_visible term)"""
        S_true = Sequence(y_truth, initial_elem = initial_element)
        degree = S_true.degree
        elem_true = S_true.n_first_elem(self.config.n_visible_terms)
        try:
            S_pred = Sequence(y_pred, initial_elem= initial_element)
            elem_pred = S_pred.n_first_elem(self.config.n_visible_terms)
            return elem_true[degree:] == elem_pred[degree:]
        except Exception as e:
                    return False


    def score_answer(self, answer, entry):
        """
        Score the predicted recursive formula (y_pred) against the true one (y_truth).
    
        Returns:
            float: A score between 0 and 1 based on correctness, simplicity, and efficiency.
        """
        initial_terms = entry.metadata["initial terms"]
        n_visible = len(entry.metadata['first elements'])
        formula_true = str(entry.answer)        
        S_true = Sequence(formula_true, initial_elem = initial_terms)
        degree_true = S_true.degree
        terms_true = S_true.n_first_elem(n_visible)

        try:
            formula_pred = str(answer)
            S_pred = Sequence(formula_pred, initial_elem= initial_terms)
            terms_pred = S_pred.n_first_elem(n_visible)
        except Exception as e:
            return 0.0

        base_score = sum(a == b for a, b in zip(terms_pred[degree_true:], terms_true[degree_true:]))/len(terms_true[degree_true:])

        if base_score < 0.5: #if half of the sample are not predicted well, then consider it as not predicted
            return 0

        degree_score = (1 + degree_true) / (1 + S_pred.degree)

        # Efficiency penalty from operator usage
        ops_pred = parse_recursive_formula(formula_pred)
        ops_true = parse_recursive_formula(formula_true)

        total_ops_true = sum(ops_true.get(op, 0) for op in ops_true)
        total_ops_pred = sum(ops_pred.get(op, 0) for op in ops_pred)

        ops_conciseness_score = (1 + total_ops_true) / (1 + total_ops_pred)

        final_score = base_score * min(1.0, degree_score * ops_conciseness_score)

        return final_score

        
    def prompt(self, metadata) -> str:
        """Turns a problem into a prompt formatted with Premise and Hypothesis"""
    
        P = (
        f"You are given a sequence of {self.config.n_visible_terms} numbers generated by a recursive formula of known degree of recursion, here equal to {metadata['degree of recursion']}.\n"
        f"The indexation of the sequence start from 0, i.e. we provide you [U0,U1,...,U{self.config.n_visible_terms - 1}]."
        "Your task is to infer the formula that defines U[n] in terms of previous values and the current index n.\n\n"
        "Instruction:\n"
        "- Use only the binary operators: +, -, *, **\n"
        )
    
        if self.config.mode == "full":
            P += (
            "- If needed, you can use the unary operators relu(), sign() or the binary one / (Euclidean division), or Mod(_,_) (modulo operator, i.e., residue of Euclidean division)\n"
            )

        P += (
        "- Reference if necessary to previous terms as U[n - 1], U[n - 2], ..., U[n - d] where d is the given degree of recursion (use exactly this format)\n"
        "- You can use \"n\" as the current index (e.g., U[n] = n)\n"
        "- You must only provide the right-hand side of the formula (i.e., f(U[n], n) such that U[n] = f(...))\n"
        """- ⚠️ This implies to not include "U[n] =" in your output.\n """
        "- The degree of recursion of your guessed formula must be inferior or equal to the one of the true formula\n\n"
        "- The sequence you are asked to induce its recursive formula have the following properties:\n"
        f"Sequence: {metadata['first elements']}\n"
        f"Degree of recurrence: {metadata['degree of recursion']}\n"
        f"Initial terms: {metadata['initial terms']}\n"
        )   

        P += (
        "\n Your provided answer must be valid for all terms n ≥ d, and must be as simple as possible."
        )
        return P


    def deduplication_key(self, problem):
        """
        The couple (formula,initial_terms) is sufficient/necessary in order to characterize the sequence
        """
        return problem.answer , problem.metadata['first elements']

#--- Count elements of formulas --- 🔢

def parse_recursive_formula(formula):
    counts = defaultdict(int)
    
    # Pattern for unitary functions: e.g., Relu(x), sign(x)
    unitary_func_pattern = re.compile(r'\b(\w+)\s*\(')
    
    # Pattern for binary operators: +, -, *, /
    binary_op_pattern = re.compile(r'([+\-*/])')
    
    # Pattern for recursive terms: U[n - 1], U[n - 2], etc.
    recursive_pattern = re.compile(r'(U\[n - \d+\])')

    # Pattern for recursive terms: U[n-1], U[n-2], etc.
    recursive_pattern = re.compile(r'(U\[n-\d+\])')
    
    # Pattern for Mod(x, y)
    mod_pattern = re.compile(r'\bMod\s*\([^,]+,\s*[^)]+\)')
    
    # --- Step 1: Count Mod(x, y)
    mod_matches = mod_pattern.findall(formula)
    counts['Mod'] += len(mod_matches)
    
    # --- Step 2: Count unitary functions (excluding Mod)
    unitary_funcs = unitary_func_pattern.findall(formula)
    for func in unitary_funcs:
        if func != 'Mod':
            counts[func] += 1
    
    # --- Step 3: Count binary operations
    binary_ops = binary_op_pattern.findall(formula)
    for op in binary_ops:
        counts[op] += 1
    
    # --- Step 4: Count recursive elements u[n-x]
    recursive_terms = recursive_pattern.findall(formula)
    for term in recursive_terms:
        counts[term] += 1
    
    # --- Step 5: Count 'n' occurrences outside recursive elements
    # Remove all u[n-x] to prevent double counting 'n'
    formula_no_recursive = recursive_pattern.sub('', formula)
    
    # Match standalone n (not part of variable names)
    n_pattern = re.compile(r'\bn\b')
    n_matches = n_pattern.findall(formula_no_recursive)
    counts['n'] += len(n_matches)
    
    return dict(counts)


# --- Simplification Module --- ♻️

# Define symbolic function objects for sign, relu, and division.
sign_sym = sp.Function('sign')
relu_sym = sp.Function('relu')
div_sym = sp.Function('/')

def Sign(x):
    x = sp.sympify(x)
    if x.is_number:
        if x > 0:
            return sp.S.One
        elif x < 0:
            return -sp.S.One
        else:
            return sp.S.Zero
    elif x.is_negative:
        return -sp.S.One
    elif x.is_positive:
        return sp.S.One
    return sign_sym(x)

def Relu(x):
    x = sp.sympify(x)
    if x.is_number:
        return x if x >= 0 else sp.S.Zero
    if x.is_nonnegative:
        return x
    if x.is_negative:
        return sp.S.Zero
    return relu_sym(x)

def safe_div(x, y):
    x = sp.sympify(x)
    y = sp.sympify(y)
    if x.is_number and y.is_number:
        return x // y
    return div_sym(x, y)

def Mod(x, y):
    x = sp.sympify(x)
    y = sp.sympify(y)
    return x % y

def convert_to_sympy(tokens, recurrence_depth = 3):
    """Convert a list of tokens to a SymPy expression with special handling"""

    expr_str = ''.join(tokens)
    n = sp.symbols('n', integer=True, nonnegative=True)
    U = sp.IndexedBase('U')   
    
    # Build a dictionary of the u_i variables dynamically.
    U_vars = {f'U{i}': U[n - i] for i in range(1, recurrence_depth+1)}

    # Local dictionary with safe operations and the dynamically generated u variables.
    local_dict = {
            "relu": Relu,  
            "mod": Mod,
            "sign": Sign,
            '/': safe_div,
                    }
    local_dict.update(U_vars)
    expr = sp.sympify(expr_str, locals=local_dict)
    return expr
