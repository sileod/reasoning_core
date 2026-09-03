import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'open_addressing_slot (draw 1 of 2)',
 'hypothesis': 'W1-010',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/open_addressing_slot',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 64225588,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _insert_slots(insert_keys, step, size):
    occupied = [False] * size
    slots = {}
    for key in insert_keys:
        start = (key * step) % size
        cell = None
        for s in range(size):
            probe = (start + s * step) % size
            if not occupied[probe]:
                occupied[probe] = True
                cell = probe
                break
        if cell is None:
            return None
        slots[key] = cell
    return slots, occupied


def _query_slot(query_key, step, size, occupied):
    start = (query_key * step) % size
    for s in range(size):
        probe = (start + s * step) % size
        if not occupied[probe]:
            return probe
    return None


@dataclass
class OpenAddressingSlotConfig(Config):
    size: int = 11
    n_inserts: int = 6

    def apply_difficulty(self, level):
        self.size = int(11 + level * 2 + level * level)
        self.n_inserts = int(4 + level)


class OpenAddressingSlot(Task):
    summary = "Insert keys under a stated open-addressing probe rule (linear probing with a constant step) and output the queried key's slot; varied table sizes, probe steps, key sets and query identities."

    config_cls = OpenAddressingSlotConfig

    def generate_entry(self):
        cfg = self.config
        size = cfg.size
        n_inserts = cfg.n_inserts
        step = random.choice([1, 2, 3])
        while True:
            insert_keys = random.sample(range(0, 200), n_inserts)
            result = _insert_slots(insert_keys, step, size)
            if result is not None:
                slots, occupied = result
                break
        query_key = random.randint(200, 400)
        answer = _query_slot(query_key, step, size, occupied)
        metadata = edict({
            "step": step,
            "size": size,
            "insert_keys": insert_keys,
            "query_key": query_key,
        })
        metadata.payload = {
            "step": step,
            "size": size,
            "insert_keys": insert_keys,
            "query_key": query_key,
        }
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        return (
            "A hash table of " + str(metadata.size) + " slots indexed 0 through "
            + str(metadata.size - 1) + " uses open addressing with linear probing: "
            "h(key) = (key * " + str(metadata.step) + ") mod " + str(metadata.size)
            + ", and to insert key k the table checks slots h(k), h(k)+" + str(metadata.step)
            + ", h(k)+2*" + str(metadata.step) + ", ... wrapping modulo "
            + str(metadata.size) + ", placing k in the first slot not yet occupied.\n\n"
            "Each of these keys is inserted in the order given, each into the first "
            "unoccupied slot its probe sequence reaches:\n"
            + ", ".join(str(k) for k in metadata.insert_keys) + "\n\n"
            "Now the key " + str(metadata.query_key) + " is probed with the same rule. "
            "Considering the slots it would find occupied by the keys above, the first "
            "slot on " + str(metadata.query_key) + "'s probe sequence that is empty is "
            "where it would be inserted.\n\n"
            "The answer is the index of that slot."
        )

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(str(answer).strip()) == int(entry.answer) else 0.0
        except Exception:
            return 0.0
