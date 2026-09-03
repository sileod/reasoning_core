import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'red_black_black_height (draw 1 of 2)',
 'hypothesis': 'W1-018',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/red_black_black_height',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 306550574,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


BLACK = 0
RED = 1

NIL = None


class RBNode:
    __slots__ = ('key', 'color', 'left', 'right', 'parent')

    def __init__(self, key):
        self.key = key
        self.color = RED
        self.left = None
        self.right = None
        self.parent = None


def is_red(n):
    return n is not None and n.color == RED


def is_black(n):
    return n is None or n.color == BLACK


class RedBlackTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        n = RBNode(key)
        if self.root is None:
            n.color = BLACK
            self.root = n
            return
        cur = self.root
        while True:
            if key < cur.key:
                if cur.left is None:
                    cur.left = n
                    n.parent = cur
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = n
                    n.parent = cur
                    break
                cur = cur.right
        self._fix(n)

    def _fix(self, n):
        while is_red(n.parent):
            p = n.parent
            g = p.parent
            if p is g.left:
                u = g.right
                if is_red(u):
                    p.color = BLACK
                    u.color = BLACK
                    g.color = RED
                    n = g
                else:
                    if n is p.right:
                        n = p
                        self._rotate_left(n)
                    n.parent.color = BLACK
                    n.parent.parent.color = RED
                    self._rotate_right(n.parent.parent)
            else:
                u = g.left
                if is_red(u):
                    p.color = BLACK
                    u.color = BLACK
                    g.color = RED
                    n = g
                else:
                    if n is p.left:
                        n = p
                        self._rotate_right(n)
                    n.parent.color = BLACK
                    n.parent.parent.color = RED
                    self._rotate_left(n.parent.parent)
        self.root.color = BLACK

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y


def _walk(node):
    if node is None:
        return 0, True
    lb, lvalid = _walk(node.left)
    rb, rvalid = _walk(node.right)
    if not (lvalid and rvalid):
        return 0, False
    if lb != rb:
        return 0, False
    if is_red(node) and (is_red(node.left) or is_red(node.right)):
        return 0, False
    return lb + (1 if node.color == BLACK else 0), True


def valid_rb(root):
    if root is None:
        return True, 0
    if root.color != BLACK:
        return False, 0
    return _walk(root)


def black_height(node):
    if node is None:
        return 0
    return black_height(node.left) + (1 if node.color == BLACK else 0)


def _serialize(node):
    if node is None:
        return '()'
    color = 'B' if node.color == BLACK else 'R'
    return f"[{node.key}{color},{_serialize(node.left)},{_serialize(node.right)}]"


def _collect(node, out):
    if node is None:
        return
    out.append(node)
    _collect(node.left, out)
    _collect(node.right, out)


class RedBlackBlackHeightConfig(Config):
    n_nodes: int = 6

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)


class RedBlackBlackHeight(Task):
    summary = "Given a valid red-black tree and node, output its black-height."

    config_cls = RedBlackBlackHeightConfig

    def generate_entry(self):
        n_nodes = self.config.n_nodes
        tree = RedBlackTree()
        keys = set()
        while len(keys) < n_nodes:
            keys.add(random.randrange(10 ** 6))
        for k in keys:
            tree.insert(k)
        ok, _ = valid_rb(tree.root)
        if not ok:
            raise RuntimeError('invalid rb tree produced')

        nodes = []
        _collect(tree.root, nodes)
        node = random.choice(nodes)
        bh = black_height(node)
        if bh < 0:
            raise RuntimeError('negative black height')

        meta = {'serialized': _serialize(tree.root),
                'node_key': node.key,
                'node_color': 'red' if node.color == RED else 'black'}
        meta['payload'] = {'serialized': meta['serialized'],
                           'node_key': meta['node_key'],
                           'node_color': meta['node_color']}
        return Entry(metadata=edict(meta), answer=str(bh))

    def render_prompt(self, metadata):
        return (f"A valid red-black tree is serialized as nested (key,color,left,right) triples where "
                f"color is B (black) or R (red) and an empty subtree is written as (), "
                f"e.g. [5B,[3R,(),()],[9B,(),()]]: {metadata.serialized}. "
                f"The node with key {metadata.node_key} has color {metadata.node_color}. "
                f"Report the black-height of the node with key {metadata.node_key}, defined as the number of "
                f"black nodes on the (unique) path from that node down to a leaf, counting the node itself if it "
                f"is black and not counting the leaf's null placeholder.\n\n"
                f"The answer is the black-height, a non-negative integer.")

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        s = str(answer).strip()
        try:
            v = int(s)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if v == int(entry.answer) else 0.0
