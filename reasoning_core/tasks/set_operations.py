import random
import numpy as np
import datetime
from num2words import num2words
from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config
import itertools
import string

### Tool functions 🛠️

def return_shuffle(domain):
    """Domain must be a collection of element convertible in list"""
    domain = list(domain)
    random.shuffle(domain)
    return domain


def random_subdomain(domain, size=None):
    """Domain must be a collection of element convertible in list, and frac must be a float between 0 and 1. \n
    return a random fraction of the domain."""
    domain = list(domain)
    subset = random.sample(domain, size)
    return return_shuffle(subset)

def create_intension(domain : list, length : int):
        """Returns a contiguous subdomain (of domain) of size length."""
        n = len(domain)
        i = np.random.randint(n-length)
        return domain[i:i+length]

def make_domains(size):
    
    NUM = [int(i) for i in range(1,size+1)]
    NUM_EN = [num2words(i,lang='en') for i in NUM]
    NUM_FR = [num2words(i,lang='fr') for i in NUM]
    
    start = (datetime.date(2020, 1, 1))
    DATES = [(start + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(size)]
    DATES_EN = [(start + datetime.timedelta(days=i)).strftime('%B %d, %Y') for i in range(size)]
    
    gen = itertools.chain.from_iterable((''.join(p) for p in itertools.product(string.ascii_lowercase, repeat=n)) for n in itertools.count(1))
    LETTERS = list(itertools.islice(gen, size))

    domains = [NUM, NUM_EN, NUM_FR, DATES, DATES_EN, LETTERS]
    return domains

def perturb_list(input_l, base_domain, n_perturbation=1):
    for _ in range(n_perturbation):
        perturbation = random.choice(['add', 'remove', 'replace'])
        lst = input_l[:]
        complementary = list(set(base_domain) - set(lst))
        if perturbation == 'add':
            new_element = random.choice(complementary)
            insert_pos = random.randint(0, len(lst))
            lst.insert(insert_pos, new_element)
        elif perturbation == 'remove' and len(lst) > 1:
            del lst[random.randint(0, len(lst) - 1)]
        elif perturbation == 'replace':
            index = random.randint(0, len(lst) - 1)
            lst[index] = random.choice(complementary)
    return (lst , perturbation)


def jaccard_similarity(set1, set2):
    return len(set1 & set2)/len(set1 | set2)
    
### Task class 🎮 🎯

@dataclass
class SetOpsConfig(Config):
    domain_size: int = 1000
    set_size: int = 10
    n_max_perturbation: int = 2
    prob_equal: float = 0.5
    def update(self, c):
        self.set_size *= 1 + c
        self.domain_size *= 1 + c
        self.n_max_perturbation *= 1 + c

        
class SetIntersection(Task):
    def __init__(self, config=SetOpsConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)
    
    def generate(self):
        chosen_domain = random.choice(self.domains)
        N=6
        set_1 = random_subdomain( chosen_domain, size=self.config.set_size) 
        others = list(set(chosen_domain) - set(set_1))
        set_2 = random.sample(others, N//2) + random.sample(set_1, N//2)
        inter = set(set_1) & set(set_2)
        meta = {}
        meta["set_1"] = return_shuffle(set_1)
        meta["set_2"] = return_shuffle(set_2)
        return Problem(metadata = meta, answer = str(inter))
     
    def prompt(self, metadata) -> str:
        return (
            f"Set1: {metadata['set_1']}\n"
            f"Set2: {metadata['set_2']}\n"
            "Only return the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}."
        )

    def score_answer(self, answer, entry):
        reference = entry['answer']
        try:
            set_pred , set_truth  = set(eval(answer)),set(eval(reference))
            if set_truth == set():
                return int(set_pred == set())
    
            return jaccard_similarity(set_pred, set_truth)
        except:
            return 0

@dataclass
class SetMissingElementConfig(SetOpsConfig):
    set_size: int = 10
    def update(self, c):
        self.set_size *= 1 + c
        self.n_max_perturbation *= 1 + c
        self.domain_size *= 1 + c

class SetMissingElement(Task):
    def __init__(self, config=SetMissingElementConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)
        
    def generate(self):
        chosen_domain = random.choice(self.domains)
        intention = create_intension(chosen_domain, self.config.set_size)
        missing_element = np.random.choice(intention[1:-1])
        intention.remove(missing_element)
        return Problem(metadata = {'element_list' : return_shuffle(intention) }, answer = str(missing_element))

    def prompt(self, metadata) -> str:
        return (
            f"Set_A: {metadata['element_list']}\n"
            "Only return the string element missing from Set_A."
        )

@dataclass
class SetEquality(Task):
    def __init__(self, config=SetOpsConfig()):
        super().__init__(config=config)
        self.domains = make_domains(self.config.domain_size)

    def generate(self):
        chosen_domain =random.choice(self.domains)
        subset = random_subdomain(chosen_domain, size = self.config.set_size)
        meta = {}
        meta["base_subset"] = return_shuffle(subset)
        subset_bis = subset
        perturbation = None
        n_pert = random.choice([k+1 for k in range(self.config.n_max_perturbation)])
        if random.random() > self.config.prob_equal:
            subset_bis , perturbation = perturb_list(subset, chosen_domain, n_pert)
        meta["subset_bis"] = return_shuffle(subset_bis) 
        meta["perturbation"] = perturbation
        return Problem(metadata = meta, answer = str((set(subset) == set(subset_bis))))

    def prompt(self, metadata) -> str:
        return (
            f"Set1: {metadata['base_subset']}\n"
            f"Set2: {metadata['subset_bis']}\n"
            "Only return True if Set1 and Set2 contain exactly the same elements, False otherwise."
        )
    

