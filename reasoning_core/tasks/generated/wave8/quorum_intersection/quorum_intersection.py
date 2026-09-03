import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload
from reasoning_core.template import stochastic_rounding as sround
from reasoning_core.utils import score_scalar


def _guaranteed_intersection(n, r, w):
    return max(0, r + w - n)


@dataclass
class QuorumIntersectionConfig(Config):
    n_lo: int = 5
    n_hi: int = 9

    def apply_difficulty(self, level):
        self.n_lo = sround(5 + level)
        self.n_hi = sround(10 + 2 * level)


class QuorumIntersection(Task):
    summary = (
        "For replica count N and read/write quorum sizes r,w, give the guaranteed "
        "minimum number of shared replicas max(0, r+w-N) between any read and write "
        "quorum; the mutual-exclusion requirement holds exactly when r+w>N."
    )
    config_cls = QuorumIntersectionConfig

    def generate_entry(self):
        n = random.randint(self.config.n_lo, self.config.n_hi)
        a = random.randint(0, n)
        if a == 0:
            r = random.randint(1, n - 1)
            w = random.randint(1, n - r)
        else:
            r = random.randint(a, n)
            w = a + n - r
        g = _guaranteed_intersection(n, r, w)
        assert g == a
        assert g >= 0
        assert g <= min(r, w)
        metadata = edict({
            "replicas": int(n),
            "read_size": int(r),
            "write_size": int(w),
            "guaranteed": int(g),
            "strictly_forced": bool(r + w > n),
        })
        metadata.payload = {
            "replicas": metadata.replicas,
            "read_size": metadata.read_size,
            "write_size": metadata.write_size,
        }
        return Entry(metadata=metadata, answer=str(g))

    def render_prompt(self, metadata):
        return (
            f"A quorum system has {metadata.replicas} replicas numbered 1..{metadata.replicas}. "
            f"A read quorum is any set of exactly {metadata.read_size} replicas; a write quorum is "
            f"any set of exactly {metadata.write_size} replicas. Every read operation must read all "
            f"replicas in some read quorum, and every write must be applied to all replicas in some "
            f"write quorum. In the worst case, a read quorum and a write quorum may share as few "
            f"replicas as possible. "
            f"Give the smallest possible number of replicas that must be shared between some read "
            f"quorum and some write quorum. The answer is a single non-negative integer. For "
            f"example, with 10 replicas, read size 6 and write size 7 the answer is 3."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'quorum_intersection (draw 1 of 2)',
 'hypothesis': 'W1-040',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/quorum_intersection',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4167519577,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
