from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class TreeReconstructionConfig(Config):
    n_nodes: int = 6
    max_depth: int = 4

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 3 * level)
        self.max_depth = sround(self.max_depth + level)


class TreeReconstruction(Task):
    summary = "Recover binary-tree roots from preorder-plus-inorder or inorder-plus-postorder traversal pairs."

    config_cls = TreeReconstructionConfig

    def _order_traversal(self, root, left, right):
        pre = []
        ino = []
        stack = []
        def preorder(i):
            if i == -1:
                return
            pre.append(i)
            preorder(left[i])
            preorder(right[i])
        def inorder(i):
            if i == -1:
                return
            inorder(left[i])
            ino.append(i)
            inorder(right[i])
        preorder(root)
        inorder(root)
        return pre, ino

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes

        while True:
            labels = random.sample(range(1, 1000), n)
            nodes = [0] + [random.randrange(0, i) for i in range(1, n)]
            children = [[] for _ in range(n)]
            for i in range(1, n):
                children[nodes[i]].append(i)
            max_deg = max(len(c) for c in children)
            if max_deg <= 2:
                break

        left = {}
        right = {}
        for i in range(n):
            if len(children[i]) == 2:
                left[i] = children[i][0]
                right[i] = children[i][1]
            elif len(children[i]) == 1:
                left[i] = children[i][0]
                right[i] = -1
            else:
                left[i] = -1
                right[i] = -1

        pre, ino = self._order_traversal(0, left, right)

        qtype = random.choice(["preinorder", "inpostorder"])

        metadata = edict({
            "n": n,
            "labels": labels,
            "qtype": qtype,
        })

        if qtype == "preinorder":
            metadata.payload = {
                "query": "A binary tree has the following preorder and inorder traversals (node indices). "
                         "Reconstruct the tree and report the label of the root node.",
                "preorder": pre,
                "inorder": ino,
            }
        else:
            pos = {}
            for idx, val in enumerate(ino):
                pos[val] = idx
            po = self._postorder(0, left, right)
            metadata.payload = {
                "query": "A binary tree has the following inorder and postorder traversals (node indices). "
                         "Reconstruct the tree and report the label of the root node.",
                "inorder": ino,
                "postorder": po,
            }

        metadata.root_label = labels[0]
        answer = str(labels[0])
        return Entry(metadata=metadata, answer=answer)

    def _postorder(self, root, left, right):
        res = []
        def post(i):
            if i == -1:
                return
            post(left[i])
            post(right[i])
            res.append(i)
        post(root)
        return res

    def render_prompt(self, metadata):
        return f"{render_payload(metadata.payload)}\n\nThe answer is the root node label (a single integer)."

    def score_answer(self, answer, entry):
        try:
            a = int(str(answer).strip())
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if a == int(entry.answer) else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add binary-tree reconstruction from traversal pairs.',
 'hypothesis': 'N10',
 'changes': 'Implement parent, child, and subtree queries after traversal '
            'reconstruction.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 359706907,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
