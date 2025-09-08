import copy
import random
import re
from unigram.solver_utils.tptp import split_clauses, run, to_tptp, extract_inferences_and_formulas

def satify_premise(x, n=3,verbose=False, background=''):
    def unslot_object(obj):
        cls = obj.__class__
        unslotted_cls = type(f"{cls.__name__}Unslotted", (cls,), {})
        new_obj = unslotted_cls.__new__(unslotted_cls)
        for name in cls.__slots__:
            if hasattr(obj, name):
                setattr(new_obj, name, getattr(obj, name))
        return new_obj
    
    x=unslot_object(x)
    R = x.rule.__class__
    """satify a cls"""
    wrap = tqdm if verbose else list
    for round in wrap(range(n)):

        # clear cache
        x.cache=dict()
        for d in x.descendants:
            d.cache=dict()
            
        y=run(to_tptp(x.dict(),background,use_hypothesis=False))
        o=y.indices
        offender_indices = [int(i.strip('p'))-1 for i in o if i.startswith('p') if i!='p0']

        if not offender_indices:
            x.y = f"fixed_{round}"
            x.o=offender_indices
            return x
      
        o=random.choice(offender_indices)
        
        c=list(x[0][0][0].children) # make mutable
        del c[o]
        x[0][0][0].children=c
        
        def drop(s,n):
            return ''.join(
                '' if (m:=re.search(r'(\d+)', l)) and (i:=int(m.group(1)))==n
                else (l if not m else l[:m.start(1)]+str(i-(i>n))+l[m.end(1):])
                for l in s.splitlines(True))
        x[0][0][0].rule.templates = {l : drop(v,5) for l,v in x[0][0][0].rule.templates.items()}
        
    x.y="not_fixed"
    x.o=offender_indices
    return x

def cat_premises(x,y):
    """cat non-setup terms"""
    x,y=copy.deepcopy((x,y))
    x[0][0][0].children+=(y[0][0][0].children)
    n=len(x[0][0][0].children)
    x[0][0][0].rule=copy.deepcopy(x[0][0][0].rule)
    wrap =  lambda s: (re.sub(r'(\d+)', r'{\1}', s) if type(s)==str else s)
    conj_tptp = lambda i: '&\n'.join([f'({i})' for i in range(i)])
    conj_eng  = lambda i: '\n'.join([f'{i}' for i in range(i)])
    x[0][0][0].rule.templates = {"eng":wrap(conj_eng(n)), "tptp":wrap(conj_tptp(n))}    
    return x