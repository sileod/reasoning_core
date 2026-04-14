import sqlite3
import random
import string
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


####### Data Structures
@dataclass
class FunctionRecord:
    """
    One row from the function DB
    """
    id: int
    library: str
    module: str
    name: str
    inputs: list[tuple[str, str]]
    output: str

    @property
    def is_leaf(self) -> bool:
        """
        Zero-argumemt functions are always valid leaves
        """ 
        return len(self.inputs) == 0

    def import_path(self)->str:
        return f"from {self.module} import {self.name}"

@dataclass
class ExpressionNode:
    """
    A node in the expression tree
    """
    func: FunctionRecord
    args: list["ExpressionNode | LeafNode"] = field(default_factory=list)

    @property
    def return_type(self) -> str:
        return self.func.output

@dataclass
class LeafNode:
    """
    A terminal value node (literal or placeholder constant)
    """
    type_str: str
    value: str
    is_literal: bool

    @property
    def return_type(self) -> str:
        return self.type_str

@dataclass
class GeneratedProgram:
    """
    The final output of one generation run
    """
    tree: dict
    code: str
    imports: list[str]



####### Primitive Literal Generators
# Types we can synthesize random literals for
PRIMITIVE_TYPES = ["int", "float", "str", "bool", "bytes"]

def _random_literal(type_str: str, rng:random.Random) -> Optional[str]:
    """
    Return a Python source literal for a primitive type, or None if not primitive
    """
    t = type_str.lower()

    if t == "int":
        return str(rng.randint(-100, 100))

    if t == "float":
        return repr(round(rng.uniform(-100.0, 100.0), 4))

    if t == "str":
        length = rng.randint(3, 8)
        return repr(''.join(rng.choices(string.ascii_lowercase, k=length)))

    if t == "bool":
        return rng.choice(["True", "False"])

    if t == "bytes":
        length = rng.randint(1, 4)
        val = bytes(rng.randint(0, 255) for _ in range(length))
        return repr(val)
    
    return None


####### Type Index
class TypeIndex:
    """
    Indexes all functions from the DB by their type.
    Strict matching: 'str' only matches 'str'
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

        ## output_type -> list of FunctionRecord
        self._by_output: dict[str, list[FunctionRecord]] = defaultdict(list)
        self._all: list[FunctionRecord] = []
        self._load()

    def _parse_inputs(self, inputs_str: str) -> list[tuple[str, str]]:
        """
        Parse 'name: type, name: type, ...' into [(bame, type), ....]
        Returns [] for empty/blank inputs
        """
        if not inputs_str or not inputs_str.strip():
            return []

        result = []
        for part in inputs_str.split(","):
            part = part.strip()
            if ":" in part:
                name, _, typ = part.partition(":")
                result.append((name.strip(), typ.strip()))
            else:
                result.append((f"arg{len(result)}", part.strip()))
        return result

    def _load(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, library, module, function_name, inputs, outputs FROM functions")
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            id_, lib, mod, name, inputs_str, output = row
            inputs = self._parse_inputs(inputs_str or "")
            output = (output or "").strip()
            if not output:
                continue
            rec = FunctionRecord(
                id=id_,
                library=lib,
                module=mod,
                name=name,
                inputs=inputs,
                output=output
            )
            self._by_output[output].append(rec)
            self._all.append(rec)

    def functions_returning(self, type_str: str) -> list[FunctionRecord]:
        """
        Return all functions whose output strictly matches type_str
        """
        return self._by_output.get(type_str, [])

    def random_function(self, rng: random.Random) -> FunctionRecord:
        """
        Pick any function at random from the full index
        """
        return rng.choice(self._all)

    def summary(self) -> dict:
        return {
            "total_functions": len(self._all),
            "unique_output_types": len(self._by_output),
            "output_type_counts": {k: len(v) for k, v in sorted(self._by_output.items(), key=lambda x: -len(x[1]))[:20]}
        }


####### Program Generator
class ProgramGenerator:
    """
    Recursively builds expression trees of typed function compositions

    Strategy (bakcward):
    1. Pick a root function at random
    2. For each of its parameters, try to find a function whose output matches the required type and recurse (depth -1)
    3. At depth 0, or when no function satisfies a type, if primitive -> random literal LeafNode, otherwise -> placeholder constant LeafNode
    4. Zero-arg functions are valid at any depth (depth-0 leaves by nature)
    """

    def __init__(self, db_path: str, max_depth: int = 3, seed: Optional[int] = None):
        self.index = TypeIndex(db_path)
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self._const_counter: dict[str, int] = defaultdict(int)

    def _fresh_const(self, type_str: str) -> str:
        """
        Generate a unique placeholder constant name
        """
        safe = type_str.replace(" ", "_").replace("|", "Or").replace("]","_")
        idx = self._const_counter[safe]
        self._const_counter[safe] += 1
        return f"CONST_{safe}_{idx}"

    def _make_leaf(self, type_str: str) -> LeafNode:
        """
        Create a leaf node for the given type
        """
        lit = _random_literal(type_str, self.rng)
        if lit is not None:
            return LeafNode(type_str=type_str, value=lit, is_literal=True)
        const_name = self._fresh_const(type_str)
        return LeafNode(type_str=type_str, value=const_name, is_literal=False)

    def _build(self, required_type: str, depth: int) -> "ExpressionNode | LeafNode":
        """
        Recursively build a node that produces 'required_type'
        """
        candidates = self.index.functions_returning(required_type)

        ## At max depth or no candidates: return a leaf
        if depth == 0 or not candidates:
            return self._make_leaf(required_type)

        ## Pick a random candidate function
        func = self.rng.choice(candidates)

        ## zero-arg functions -> ExpressionNode with no args (valid at any depth)
        if func.is_leaf:
            return ExpressionNode(func=func, args=[])

        ## recursively satisfy each argument
        args = []
        for _param_name, param_type in func.inputs:
            arg_node = self._build(param_type, depth - 1)
            args.append(arg_node)

        return ExpressionNode(func=func, args=args)

    def generate(self, target_type: Optional[str] = None) -> GeneratedProgram:
        """
        Generate one program

        Args: 
            target_type: if given, the root function must return this type. 
                            If None, a random function is chosen as root.
        """
        ## reset constant counter for fresh names each generation
        self._const_counter.clear()

        if target_type:
            candidates = self.index.functions_returning(target_type)
            if not candidates:
                raise ValueError(f"No functions found returning type '{target_type}'")
            root_func = self.rng.choice(candidates)
        else:
            root_func = self.index.random_function(self.rng)

        ## build the root node
        if root_func.is_leaf:
            root = ExpressionNode(func=root_func, args=[])
        else:
            args = []
            for _param_name, param_type in root_func.inputs:
                arg_node = self._build(param_type, self.max_depth - 1)
                args.append(arg_node)
            root = ExpressionNode(func=root_func, args=args)

        ## Emit outputs
        imports: list[str] = []
        tree = _tree_to_dict(root)
        code = _emit_code(root, imports)

        ## Deduplicate imports preserving order
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)

        full_code = "\n".join(unique_imports) + "\n\n" + code if unique_imports else code

        return GeneratedProgram(tree=tree, code=full_code, imports=unique_imports)

###### Tree + Dict (JSON Structure)
def _tree_to_dict(node: "ExpressionNode | LeafNode") -> dict:
    if isinstance(node, LeafNode):
        return {
            "kind": "leaf",
            "type": node.type_str,
            "value": node.value,
            "is_literal": node.is_literal
        }
    assert isinstance(node, ExpressionNode)
    return {
        "kind": "call",
        "function": node.func.name,
        "module": node.func.module,
        "return_type": node.func.output,
        "args": [
            {
                "param": node.func.inputs[i][0] if i < len(node.func.inputs) else f"args{i}",
                "expected_type": node.func.inputs[i][1] if i < len(node.func.inputs) else "?",
                "node": _tree_to_dict(arg)
            }
            for i, arg in enumerate(node.args)
        ],
    }

###### Tree + Python code
def _emit_expr(node: "ExpressionNode | LeafNode", imports: list[str]) -> str:
    """
    Recursively emit a Python expression string for a node
    """
    if isinstance(node, LeafNode):
        return node.value

    assert isinstance(node, ExpressionNode)
    imports.append(node.func.import_path())

    arg_exprs = [_emit_expr(arg, imports) for arg in node.args]

    if node.func.inputs:
        named_args = ", ".join(
            f"{node.func.inputs[i][0]}={expr}" if i < len(node.func.inputs) else expr for i, expr in enumerate(arg_exprs)
        )
    else:
        named_args = ""

    return f"{node.func.name}({named_args})"


def _collect_placeholder_counts(node: "ExpressionNode | LeafNode") -> list[LeafNode]:
    """
    Walk tree and collect all non-literal leaf nodes
    """
    if isinstance(node, LeafNode):
        return [node] if not node.is_literal else []
    
    results = []
    for arg in node.args:
        results.extend(_collect_placeholder_counts(arg))
    
    return results


def _emit_code(root: "ExpressionNode | LeafNode", imports: list[str]) -> str:
    """
    Emit a full python snippet:
    1. Placeholder constant declarations (with type comments)
    2. result = <expression>
    3. print(result)
    """

    ## Collect placeholder constants first (before emitting expr with populates imports)
    consts = _collect_placeholder_counts(root)

    expr = _emit_expr(root, imports)

    lines = []

    if consts:
        lines.append("# - Placeholder constants (replace with real values) -")
        seen_consts = set()
        for leaf in consts:
            if leaf.value not in seen_consts:
                lines.append(f"{leaf.value} = None       # expected type {leaf.type_str}")
                seen_consts.add(leaf.value)
        lines.append("")

    lines.append(f"result = {expr}")
    lines.append(f"print(result)")

    return "\n".join(lines)

##### DEMO
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proceural program generator from typed function DB")
    parser.add_argument("--db", default="functions.db", help="Path to functions.db")
    parser.add_argument("--depth", type=int, default=3, help="Max composition depth")
    parser.add_argument("--n", type=int, default=3, help="Number of programs to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")
    parser.add_argument("--target-type", default=None, help="Target return type for root function")
    parser.add_argument("--summary", action="store_true", help="Print DB summary and exit")
    args = parser.parse_args()

    gen = ProgramGenerator(db_path=args.db, max_depth=args.depth, seed=args.seed)

    if args.summary:
        s = gen.index.summary()
        print(f"Total functions: {s['total_functions']}")
        print(f"Unique ret types: {s['unique_output_types']}")
        print(f"\nTop output types:")
        for t, c in s["output_type_counts"].items():
            print(f"{t:30s} {c}")
        exit(0)

    for i in range(args.n):
        print(f"\n{'='*60}")
        print(f" PROGRAM {i+1}")
        print(f"{'='*60}")
        prog = gen.generate(target_type=args.target_type)

        print(f"\n-----Python Code------")
        print(prog.code)

        print(f"\n-----Expression Tree (JSON) ------")
        print(json.dumps(prog.tree, indent=2))