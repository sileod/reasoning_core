from reasoning_core.template import Task, Entry, Config
from contextlib import chdir
from dataclasses import dataclass
from tempfile import TemporaryDirectory
try:
    import reasoning_gym
except ImportError:
    reasoning_gym = None
import random
import json

# Curated reasoning-gym task subset used for per-task influence sweeps.
# (Kept here as a plain list rather than a separate JSON data file.)
RGYM_TASKS = [
    'ab', 'acre', 'advanced_geometry', 'aiw', 'arc_1d', 'arc_agi', 'base_conversion',
    'basic_arithmetic', 'bf', 'binary_alternation', 'binary_matrix', 'bitwise_arithmetic', 'boxnet',
    'caesar_cipher', 'calendar_arithmetic', 'chain_sum', 'circuit_logic', 'codeio',
    'color_cube_rotation', 'complex_arithmetic', 'count_bits', 'count_primes', 'countdown',
    'course_schedule', 'cryptarithm', 'decimal_arithmetic', 'decimal_chain_sum', 'dice',
    'emoji_mystery', 'family_relationships', 'figlet_font', 'fraction_simplification', 'futoshiki',
    'game_of_life', 'game_of_life_halting', 'gcd', 'graph_color', 'group_anagrams', 'gsm_symbolic',
    'intermediate_integration', 'isomorphic_strings', 'jugs', 'knights_knaves', 'largest_island',
    'lcm', 'leg_counting', 'letter_counting', 'letter_jumble', 'list_functions', 'mahjong_puzzle',
    'manipulate_matrix', 'maze', 'mini_sudoku', 'n_queens', 'needle_haystack', 'number_filtering',
    'number_format', 'number_sequence', 'number_sorting', 'palindrome_generation',
    'palindrome_partitioning', 'polynomial_equations', 'polynomial_multiplication', 'pool_matrix',
    'power_function', 'prime_factorization', 'products', 'propositional_logic', 'puzzle24',
    'quantum_lock', 'ransom_note', 'rearc', 'rectangle_count', 'rotate_matrix', 'rotten_oranges',
    'rubiks_cube', 'rush_hour', 'self_reference', 'sentence_reordering', 'shortest_path',
    'simple_equations', 'simple_geometry', 'simple_integration', 'sokoban', 'spell_backward',
    'spiral_matrix', 'string_insertion', 'string_manipulation', 'string_splitting',
    'string_synthesis', 'sudoku', 'syllogism', 'time_intervals', 'tower_of_hanoi', 'tsumego',
    'word_ladder', 'word_sequence_reversal', 'word_sorting', 'zebra_puzzles',
]


@dataclass
class RGConfig(Config):
    rg_task: str = ""
    rg_level: int = 1

    def apply_difficulty(self, level):
        self.rg_level += level

class Reasoning_Gym(Task):
    summary = "Interface with diverse reasoning datasets generated via reasoning-gym."
    def __init__(self, config=None):
        if reasoning_gym is None:
            raise ImportError("reasoning_gym is not installed.")
        self.datasets = [d for d in reasoning_gym.factory.DATASETS.keys() if d != 'composite']
        super().__init__(config=config or RGConfig())

    def generate_entry(self):
        d = self.config.rg_task or random.choice(self.datasets)
        t, c_cls = reasoning_gym.factory.DATASETS[d]

        if d in reasoning_gym.factory.CURRICULA:
            cl = reasoning_gym.factory.CURRICULA[d]()
            native_max_level = max(len(attr.levels) - 1 for attr in cl.attributes.values())
            cl.set_global_level(min(int(self.config.rg_level), native_max_level))
            c = cl.generate_configuration()
            max_level = max(0, native_max_level - 1)
            effective_level = min(max(0, int(self.config.rg_level) - 1), max_level)
        else:
            c = c_cls()
            effective_level = max_level = 0

        if d == "codeio":
            with TemporaryDirectory() as tmp, chdir(tmp):
                entry = t(c)[0]
        else:
            entry = t(c)[0]
        meta = entry['metadata'] | {
            "task_name": f"RG.{d}",
            "source_collection": "reasoning_gym",
            "source_task": d,
            "effective_level": effective_level,
            "max_level": max_level,
            "_question": entry['question'],
        }
        return Entry(json.loads(json.dumps(meta, default=str)), str(entry['answer']))

    def score_answer(self, answer, entry):
        sd = (entry['metadata'].get('source_task') or entry['metadata'].get('_source_task')
              or entry['metadata'].get('source_dataset'))
        scorer = reasoning_gym.get_score_answer_fn(sd)
        try:
            score = scorer(answer,entry)
        except Exception as e:
            print(f"Error scoring, T={entry['metadata']['task_name']} answer: {e}")
            score = 0
        return score

    def render_prompt(self, metadata):
        return metadata._question


def as_reasoning_gym(task_cls):
    """Adapt a core Task class to reasoning-gym's finite dataset interface."""
    from reasoning_gym.dataset import ProceduralDataset
    from easydict import EasyDict
    from reasoning_core import score_answer

    class CoreDataset(ProceduralDataset):
        def __init__(self, config):
            super().__init__(config, seed=config.seed, size=500 if config.size is None else config.size)
            self.task = task_cls(config=config)

        def __getitem__(self, index):
            if not 0 <= index < self.size:
                raise IndexError(index)
            return self.task[index]

        def score_answer(self, answer, entry):
            return score_answer(answer, EasyDict(entry))

    CoreDataset.__name__ = task_cls.__name__ + 'Dataset'
    return CoreDataset


def register_tasks(task_names=None):
    """Register selected tasks without constructing every generator."""
    import reasoning_gym
    from reasoning_core import DATASETS

    for name in DATASETS if task_names is None else task_names:
        if name not in reasoning_gym.factory.DATASETS:
            task_cls = DATASETS[name]._resolved
            reasoning_gym.register_dataset(
                name, as_reasoning_gym(task_cls), task_cls.config_cls or type(task_cls().config),
            )
