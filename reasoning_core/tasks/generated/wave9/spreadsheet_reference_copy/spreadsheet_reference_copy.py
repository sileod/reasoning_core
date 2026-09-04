import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'spreadsheet_reference_copy (draw 1 of 1)',
 'hypothesis': 'HV-036',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/spreadsheet_reference_copy',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3518700511,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _num_to_letters(n):
    name = ""
    n = n + 1
    while n:
        n, r = divmod(n - 1, 26)
        name = chr(65 + r) + name
    return name


def _render_ref(ref):
    abs_col, col, abs_row, row = ref
    colstr = ("$" if abs_col else "") + _num_to_letters(col)
    rowstr = ("$" if abs_row else "") + str(row)
    return colstr + rowstr


def _shift_ref(ref, dc, dr):
    abs_col, col, abs_row, row = ref
    ncol = col if abs_col else col + dc
    nrow = row if abs_row else row + dr
    return (abs_col, ncol, abs_row, nrow)


def _render_node(node):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "ref":
        return _render_ref(node[1])
    return "(" + _render_node(node[2]) + node[1] + _render_node(node[3]) + ")"


def _transform_node(node, dc, dr):
    kind = node[0]
    if kind == "num":
        return node
    if kind == "ref":
        return ("ref", _shift_ref(node[1], dc, dr))
    return ("binop", node[1], _transform_node(node[2], dc, dr),
            _transform_node(node[3], dc, dr))


def _all_refs_valid(node):
    if node[0] == "ref":
        abs_col, col, abs_row, row = node[1]
        return col >= 0 and row >= 1
    if node[0] == "binop":
        return _all_refs_valid(node[2]) and _all_refs_valid(node[3])
    return True


@dataclass
class SpreadsheetCopyConfig(Config):
    expr_size: int = 1
    max_col: int = 4
    max_row: int = 6

    def apply_difficulty(self, level):
        self.expr_size = sround(self.expr_size + level)
        self.max_col = sround(self.max_col + level)
        self.max_row = sround(self.max_row + level)


class SpreadsheetReferenceCopy(Task):
    summary = ("Copy arithmetic spreadsheet formulas across rows and columns, "
               "shifting relative references while leaving $-marked absolute and "
               "mixed references fixed, and return the resulting formula.")
    config_cls = SpreadsheetCopyConfig
    task_version = 2

    def _build(self, cfg):
        def make_term():
            if random.random() < 0.15:
                return ("num", str(random.randint(1, 50)))
            abs_col = random.random() < 0.3
            abs_row = random.random() < 0.3
            return ("ref", (abs_col, random.randint(0, cfg.max_col),
                            abs_row, random.randint(1, cfg.max_row)))

        def build(n):
            if n <= 0:
                return make_term()
            left = random.randint(0, n - 1)
            op = random.choice(["+", "-", "*"])
            return ("binop", op, build(left), build(n - 1 - left))

        return build(cfg.expr_size)

    def generate_entry(self):
        cfg = self.config
        for _ in range(200):
            root = self._build(cfg)
            src_col = random.randint(0, cfg.max_col)
            src_row = random.randint(1, cfg.max_row)
            dc = random.randint(-2, 2)
            dr = random.randint(-2, 2)
            if dc == 0 and dr == 0:
                continue
            shifted = _transform_node(root, dc, dr)
            if not _all_refs_valid(shifted):
                continue
            dst_col = src_col + dc
            dst_row = src_row + dr
            orig = "=" + _render_node(root)
            copied = "=" + _render_node(shifted)
            if copied == orig:
                continue
            src_cell = _num_to_letters(src_col) + str(src_row)
            dst_cell = _num_to_letters(dst_col) + str(dst_row)
            metadata = edict({
                "payload": {
                    "Original formula": orig,
                    "Source cell": src_cell,
                    "Destination cell": dst_cell,
                }
            })
            return Entry(metadata=metadata, answer=copied)
        raise RuntimeError("spreadsheet_reference_copy: could not build a distinct copy")

    def render_prompt(self, metadata):
        return (
            "A spreadsheet cell contains a formula. The formula uses cell references; a "
            "reference is made absolute in a dimension when that dimension is preceded by "
            "a dollar sign. When the formula is copied to another cell, each relative "
            "dimension shifts by the same number of rows and columns that separate the "
            "destination cell from the source cell, while an absolute dimension stays "
            "fixed. Give the formula exactly as it reads after copying to the "
            "destination cell.\n\n"
            + render_payload(metadata.payload) +
            "\n\nThe answer is the copied formula, written without spaces."
        )

    def score_answer(self, answer, entry):
        gold = str(entry.answer).replace(" ", "")
        got = str(answer).replace(" ", "")
        return 1.0 if got == gold else 0.0
