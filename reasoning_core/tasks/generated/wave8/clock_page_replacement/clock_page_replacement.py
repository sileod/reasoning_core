"""Clock (second-chance) page replacement: given state + a fault, choose the eviction victim."""

import random

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar


TASK_META = {'parent_source_id': None,
 'idea': 'clock_page_replacement (draw 1 of 2)',
 'hypothesis': 'W1-045',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/clock_page_replacement',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2635584578,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def clock_victim(pages, bits, hand):
    """Run the Clock (second-chance) algorithm and return the victim frame index.

    Starting at `hand`, walk circularly. If a frame's reference bit is 1, give it
    a second chance: clear its bit to 0 and advance. The first frame whose bit is
    0 is evicted. Generation guarantees at least one 0 bit exists.
    """
    n = len(pages)
    bits = list(bits)
    hand = hand % n
    for _ in range(n + 1):
        if bits[hand] == 0:
            return hand
        bits[hand] = 0
        hand = (hand + 1) % n
    raise RuntimeError("no 0-bit found in clock scan")


class ClockPageReplacementConfig(Config):
    n_frames: int = 5
    max_val: int = 20

    def apply_difficulty(self, level):
        self.n_frames = int(5 + level)
        self.max_val = 20 + 6 * level


class ClockPageReplacement(Task):
    summary = (
        "Given a Clock (second-chance) page-replacement state (page frames with "
        "reference bits, clock hand pointer) and a page fault, output the single "
        "frame index chosen for eviction, varied across frame counts, hand positions, "
        "and reference-bit patterns."
    )
    config_cls = ClockPageReplacementConfig

    def generate_entry(self):
        n = self.config.n_frames
        max_val = self.config.max_val

        resident = random.sample(range(max_val), n)
        pages = list(resident)

        bits = [random.randint(0, 1) for _ in range(n)]
        if all(b == 1 for b in bits):
            bits[random.randrange(n)] = 0

        hand = random.randrange(n)

        while True:
            req = random.randrange(max_val + n)
            if req not in resident:
                break

        victim = clock_victim(pages, bits, hand)
        assert 0 <= victim < n

        meta = edict({
            "n_frames": int(n),
            "pages": pages,
            "bits": bits,
            "hand": int(hand),
            "request": int(req),
        })
        meta.payload = {
            "frames": [{"page": pages[i], "refbit": bits[i]} for i in range(n)],
            "hand": int(hand),
            "request": int(req),
        }
        return Entry(metadata=meta, answer=str(int(victim)))

    def render_prompt(self, metadata):
        frames_str = "; ".join(
            f"frame {i}: page {f['page']} (refbit {f['refbit']})"
            for i, f in enumerate(metadata.payload["frames"])
        )
        return (
            f"A system uses Clock (second-chance) page replacement. The current state:\n"
            f"{frames_str}\n"
            f"The clock hand is at frame {metadata.payload['hand']}.\n"
            f"A page fault occurs for page {metadata.payload['request']}.\n"
            f"Starting from the hand, scan frames in circular order, skipping frames "
            f"whose reference bit is 1 (clearing that bit to 0) and evicting the first "
            f"frame whose reference bit is 0.\n"
            f"The answer is the index of the frame evicted, as a single integer."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)
