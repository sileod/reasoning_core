import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


def build_poly(degree, rng):
    while True:
        bits = [1] + [rng.randint(0, 1) for _ in range(degree - 1)] + [1]
        poly = 0
        for b in bits:
            poly = (poly << 1) | b
        if poly != 0:
            return poly, bits


def crc_remainder(msg_bits, poly_bits):
    width = len(poly_bits) - 1
    reg = msg_bits + [0] * width
    for i in range(len(msg_bits)):
        if reg[i] == 1:
            for j in range(len(poly_bits)):
                reg[i + j] ^= poly_bits[j]
    return reg[len(msg_bits):]


def poly_to_string(bits):
    degree = len(bits) - 1
    terms = []
    for i, b in enumerate(bits):
        if not b:
            continue
        d = degree - i
        if d == 0:
            terms.append("1")
        elif d == 1:
            terms.append("x")
        else:
            terms.append("x^{}".format(d))
    return " + ".join(terms) if terms else "0"


def bits_to_string(bits):
    return "".join(str(b) for b in bits)


@dataclass
class CrcRemainderConfig(Config):
    msg_len: int = 12
    width: int = 5

    def apply_difficulty(self, level):
        self.msg_len = sround(self.msg_len + 2 * level)
        self.width = sround(self.width + level // 3)


class CrcRemainder(Task):
    summary = ("Given a message bitstring and a generator polynomial over GF(2), "
               "compute the CRC remainder at varied polynomial widths and message "
               "lengths, one remainder bitstring per instance.")
    config_cls = CrcRemainderConfig
    task_version = 2

    def generate_entry(self):
        rng = random
        width = int(self.config.width)
        degree = width
        msg_len = int(self.config.msg_len)
        if degree >= msg_len:
            msg_len = degree + 1

        message = [rng.randint(0, 1) for _ in range(msg_len)]
        _, poly_bits = build_poly(degree, rng)
        rem = crc_remainder(message, poly_bits)
        rem_str = bits_to_string(rem)
        msg_str = bits_to_string(message)
        poly_str = poly_to_string(poly_bits)

        metadata = edict({
            "message": msg_str,
            "polynomial": poly_str,
            "width": degree,
        })
        metadata.payload = {
            "message": metadata.message,
            "polynomial": metadata.polynomial,
        }
        return Entry(metadata=metadata, answer=rem_str)

    def render_prompt(self, metadata):
        return (
            "A CRC (cyclic redundancy check) remainder is computed over GF(2) by "
            "appending zeros and dividing the message polynomial by the generator "
            "polynomial, keeping the remainder. Given the message bitstring "
            "{message} and the generator polynomial {polynomial} (listed in order "
            "of descending degree, from x^k down to x^0), compute the CRC remainder "
            "bitstring. The answer is a bitstring of length equal to the polynomial "
            "degree, most-significant bit first.".format(
                message=metadata.message, polynomial=metadata.polynomial)
        )

    def score_answer(self, answer, entry):
        truth = entry.answer
        if not isinstance(answer, str):
            return 0.0
        cleaned = "".join(ch for ch in answer if ch in "01")
        return 1.0 if cleaned == truth else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'crc_remainder (draw 1 of 2)',
 'hypothesis': 'W1-068',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/crc_remainder',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1245131062,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
