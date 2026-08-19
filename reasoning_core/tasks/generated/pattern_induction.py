import random
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Program:
    lanes: int
    steps: tuple
    repeat: int
    turn: int


def _step_values(alphabet_size, max_step):
    values, seen = [], set()
    for distance in range(1, max_step + 1):
        for step in (distance, -distance):
            residue = step % alphabet_size
            if residue not in seen:
                values.append(step)
                seen.add(residue)
    return tuple(values)


@lru_cache(maxsize=None)
def _program_bank(alphabet_size, max_lanes, max_step, max_repeat, max_turn_period):
    steps = _step_values(alphabet_size, max_step)
    turns = (0,) + tuple(range(2, max_turn_period + 1))
    return tuple(
        Program(lanes, lane_steps, repeat, turn)
        for lanes in range(1, max_lanes + 1)
        for lane_steps in product(steps, repeat=lanes)
        for repeat in range(1, max_repeat + 1)
        for turn in turns
    )


def _walk(occurrence, turn):
    if not turn:
        return occurrence
    phase = occurrence % (2 * turn)
    return phase if phase <= turn else 2 * turn - phase


def _emit(program, offsets, position, alphabet_size):
    latent = position // program.repeat
    lane = latent % program.lanes
    occurrence = latent // program.lanes
    return (
        offsets[lane]
        + program.steps[lane] * _walk(occurrence, program.turn)
    ) % alphabet_size


def _sequence(program, offsets, length, alphabet_size):
    return [_emit(program, offsets, i, alphabet_size) for i in range(length)]


def _fit_offsets(program, sequence, alphabet_size):
    offsets = [None] * program.lanes
    for position, value in enumerate(sequence):
        latent = position // program.repeat
        lane = latent % program.lanes
        occurrence = latent // program.lanes
        required = (
            value - program.steps[lane] * _walk(occurrence, program.turn)
        ) % alphabet_size
        if offsets[lane] is None:
            offsets[lane] = required
        elif offsets[lane] != required:
            return None
    return tuple(offsets)


def _continuations(program, prefix, horizon, alphabet_size):
    fitted = _fit_offsets(program, prefix, alphabet_size)
    if fitted is None:
        return set()
    unknown = [i for i, value in enumerate(fitted) if value is None]
    continuations = set()
    for values in product(range(alphabet_size), repeat=len(unknown)):
        offsets = list(fitted)
        for lane, value in zip(unknown, values):
            offsets[lane] = value
        continuations.add(
            tuple(
                _emit(program, offsets, position, alphabet_size)
                for position in range(len(prefix), len(prefix) + horizon)
            )
        )
    return continuations


def _survivors(bank, query_prefix, demonstrations, alphabet_size):
    return [
        program
        for program in bank
        if _fit_offsets(program, query_prefix, alphabet_size) is not None
        and all(_fit_offsets(program, demo, alphabet_size) is not None for demo in demonstrations)
    ]


def _answer_set(programs, query_prefix, horizon, alphabet_size):
    answers = set()
    for program in programs:
        answers.update(_continuations(program, query_prefix, horizon, alphabet_size))
    return answers


def _letters(values):
    return " ".join(ALPHABET[value] for value in values)


@dataclass
class PatternInductionConfig(Config):
    alphabet_size: int = 6
    max_lanes: int = 2
    max_step: int = 1
    max_repeat: int = 2
    max_turn_period: int = 3
    num_demonstrations: int = 1
    demo_prefix_len: int = 7
    demo_horizon: int = 3
    query_prefix_len: int = 5
    max_query_prefix_len: int = 12
    prediction_horizon: int = 2
    max_generation_attempts: int = 200

    def apply_difficulty(self, level):
        self.alphabet_size = min(10, sround(self.alphabet_size + 0.5 * level))
        self.max_lanes = min(3, sround(self.max_lanes + 0.25 * level))
        self.max_step = min(3, sround(self.max_step + 0.35 * level))
        self.max_repeat = min(3, sround(self.max_repeat + 0.15 * level))
        self.max_turn_period = min(5, sround(self.max_turn_period + 0.35 * level))
        self.demo_prefix_len = sround(self.demo_prefix_len + 0.6 * level)
        self.max_query_prefix_len = sround(self.max_query_prefix_len + level)
        self.prediction_horizon = sround(self.prediction_horizon + 0.35 * level)


class PatternInduction(Task):
    summary = "Infer a shared symbolic sequence rule from examples and predict a uniquely determined continuation."
    config_cls = PatternInductionConfig

    def generate_entry(self):
        k = min(len(ALPHABET), max(3, int(self.config.alphabet_size)))
        max_lanes = max(1, int(self.config.max_lanes))
        max_step = max(1, int(self.config.max_step))
        max_repeat = max(1, int(self.config.max_repeat))
        max_turn = max(2, int(self.config.max_turn_period))
        n_demo = max(0, int(self.config.num_demonstrations))
        demo_prefix_len = max(2, int(self.config.demo_prefix_len))
        demo_horizon = max(1, int(self.config.demo_horizon))
        query_min = max(2, int(self.config.query_prefix_len))
        query_max = max(query_min, int(self.config.max_query_prefix_len))
        horizon = max(1, int(self.config.prediction_horizon))
        bank = _program_bank(k, max_lanes, max_step, max_repeat, max_turn)

        for _ in range(int(self.config.max_generation_attempts)):
            hidden = random.choice(bank)
            demonstrations = []
            rendered_demos = []
            for _ in range(n_demo):
                offsets = tuple(random.randrange(k) for _ in range(hidden.lanes))
                full = _sequence(hidden, offsets, demo_prefix_len + demo_horizon, k)
                if len(set(full)) < min(3, k):
                    break
                demonstrations.append(full)
                rendered_demos.append(
                    {
                        "prefix": full[:demo_prefix_len],
                        "continuation": full[demo_prefix_len:],
                    }
                )
            if len(demonstrations) != n_demo:
                continue

            offsets = tuple(random.randrange(k) for _ in range(hidden.lanes))
            query_full = _sequence(hidden, offsets, query_max + horizon, k)
            accepted = None
            for prefix_len in range(query_min, query_max + 1):
                prefix = query_full[:prefix_len]
                gold = tuple(query_full[prefix_len : prefix_len + horizon])
                if len(set(prefix)) < min(3, k) or len(set(gold)) < min(2, horizon):
                    continue

                query_only = _survivors(bank, prefix, [], k)
                query_only_answers = _answer_set(query_only, prefix, horizon, k)
                survivors = _survivors(bank, prefix, demonstrations, k)
                answers = _answer_set(survivors, prefix, horizon, k)
                if answers != {gold}:
                    continue
                if n_demo and len(query_only_answers) <= 1:
                    continue
                accepted = (
                    prefix,
                    gold,
                    survivors,
                    query_only,
                    query_only_answers,
                    prefix_len,
                )
                break

            if accepted is None:
                continue

            prefix, gold, survivors, query_only, query_only_answers, prefix_len = accepted
            metadata = edict(
                alphabet=ALPHABET[:k],
                max_lanes=max_lanes,
                allowed_steps=list(_step_values(k, max_step)),
                max_repeat=max_repeat,
                max_turn_period=max_turn,
                demonstrations=rendered_demos,
                query=prefix,
                prediction_horizon=horizon,
                hidden_program={
                    "lanes": hidden.lanes,
                    "steps": list(hidden.steps),
                    "repeat": hidden.repeat,
                    "turn": hidden.turn,
                },
                survivor_count=len(survivors),
                query_only_survivor_count=len(query_only),
                query_only_answer_count=len(query_only_answers),
                query_prefix_len=prefix_len,
            )
            return Entry(metadata=metadata, answer=_letters(gold))
        raise RuntimeError("PatternInduction: could not build a predictively unique instance")

    def render_prompt(self, m):
        steps = ", ".join(f"{s:+d}" for s in m.allowed_steps)
        turns = ", ".join(map(str, range(2, m.max_turn_period + 1)))
        lines = [
            "Infer one shared rule and continue the query.",
            f"Letters are cyclic in this order: {' '.join(m.alphabet)}.",
            "Allowed rule family:",
            f"- Interleave m lanes, with m in 1..{m.max_lanes}. Before repetition, latent positions visit lanes 0,1,...,m-1 cyclically.",
            f"- Lane j has one fixed cyclic step s_j in {{{steps}}}. Rows share m and all s_j, but may start from different letters.",
            f"- Direction is either straight, or uses one shared turn period p in {{{turns}}}. Straight uses occurrence multipliers 0,1,2,...; turn p uses 0,1,...,p,p-1,...,1,0,... periodically.",
            f"- Repeat every emitted letter r times, with r in 1..{m.max_repeat}. Rows share r and the turn choice.",
        ]
        if m.demonstrations:
            lines.append("Examples:")
            for i, demo in enumerate(m.demonstrations, 1):
                lines.append(
                    f"{i}. {_letters(demo['prefix'])} -> {_letters(demo['continuation'])}"
                )
        lines += [
            f"Query: {_letters(m.query)}",
            f"The answer is the next {m.prediction_horizon} letters, space-separated.",
        ]
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        text = str(answer).strip().upper()
        parts = [part for part in re.split(r"[\s,]+", text) if part]
        expected = entry.answer.split()
        return float(len(parts) == len(expected) and all(re.fullmatch(r"[A-Z]", part) for part in parts) and parts == expected)

    def balancing_key(self, problem):
        hidden = problem.metadata.hidden_program
        return hidden["lanes"], hidden["repeat"], bool(hidden["turn"])
