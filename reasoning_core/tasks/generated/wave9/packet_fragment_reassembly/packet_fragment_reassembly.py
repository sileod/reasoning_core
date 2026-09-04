"""Packet fragment reassembly task.

Generate a payload message that is fragmented into chunks, each carrying an
explicit (index, offset) position or a base offset. Some fragments are missing.
The model must reconstruct the full payload under explicit overlap and
missing-fragment rules, or report the error state when reconstruction is
impossible / incomplete.

The answer is a canonical string: either the reconstructed payload (with a
stable representation of any remaining gaps) or an explicit error token.
"""

import random
import re

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'packet_fragment_reassembly (draw 1 of 1)',
 'hypothesis': 'HV-075',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/packet_fragment_reassembly',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2968879882,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


ALPHABET = "ABCDEFGH"


def _build_payload(length, rng):
    """Build a payload string of the given length using the small alphabet."""
    return "".join(rng.choice(ALPHABET) for _ in range(length))


class PacketFragmentReassemblyConfig(Config):
    min_payload: int = 12
    max_payload: int = 26
    n_fragments: int = 5
    overlap_ms: int = 3
    missing_prob: float = 0.2

    def apply_difficulty(self, level):
        self.min_payload = sround(self.min_payload + level * 2)
        self.max_payload = sround(self.max_payload + level * 3)
        self.n_fragments = sround(self.n_fragments + level)
        self.missing_prob = min(0.5, 0.2 + 0.05 * level)


class _Frag:
    __slots__ = ("index", "offset", "data", "missing")

    def __init__(self, index, offset, data, missing):
        self.index = index
        self.offset = offset
        self.data = data
        self.missing = missing


def _place_fragments(payload, n_fragments, overlap_ms, rng):
    """Place n_fragments overlapping fragments covering the payload.

    Fragments advance across the payload left-to-right so the running window
    tiles the whole message and the final fragment always reaches the end.
    Overlaps agree with the true payload (each fragment is an exact slice).
    Occasionally a fragment is lost or a small gap is skipped, leaving uncovered
    positions that make full reconstruction impossible. Returns list of _Frag.
    """
    L = len(payload)
    span = max(1, L / max(1, n_fragments))
    frags = []
    placed = []
    covered = 0
    gap_prob = 0.12
    lost_prob = 0.14
    for i in range(n_fragments):
        if covered >= L:
            # reached the end; pad with short redundant fragments at the tail
            start = max(0, L - rng.randint(1, max(1, int(span)) + overlap_ms))
            end = L
            width = end - start
        else:
            start = covered
            gap = rng.random() < gap_prob and i > 0 and start < L - 2
            if gap and start < L - 2:
                start = min(L - 2, start + rng.randint(1, max(1, int(span // 2))))
            end = min(L, start + rng.randint(max(1, int(span * 0.8)),
                                             max(1, int(span * 1.6)) + overlap_ms))
            if i == n_fragments - 1:
                end = L  # last fragment always reaches the end
            width = end - start
        data = payload[start:end]
        missing = rng.random() < lost_prob and i < n_fragments - 1
        frags.append(_Frag(i, start, data, missing))
        if not missing:
            placed.append((start, end))
        covered = max(covered, min(L, end + rng.randint(0, overlap_ms)))
    return frags, placed


def _coverage(placed, L):
    cov = [False] * L
    for (s, e) in placed:
        for p in range(s, min(e, L)):
            cov[p] = True
    return cov


def _can_fully_reassemble(placed, L):
    return all(_coverage(placed, L))


def _answer_string(payload, placed, L, reassemblable):
    """Canonical answer.

    If the (non-missing) fragments fully cover the payload with consistent
    offsets, the answer is the payload itself.

    Otherwise, report error state: a compact representation of the merged
    gaps (which positions are missing). Spelled as MISSING followed by the
    sorted list of uncovered intervals, or FULL errors as a compact form.

    Since gaps only get larger (missing fragments cannot add coverage), being
    fully covered is exactly the reachable-good case. If not fully covered, the
    reconstruction is incomplete and we emit the error state listing uncovered
    intervals.
    """
    if reassemblable:
        return payload
    cov = _coverage(placed, L)
    # build uncovered intervals
    intervals = []
    i = 0
    while i < L:
        if not cov[i]:
            j = i
            while j < L and not cov[j]:
                j += 1
            intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    # compact: MISSING:<a-b,c-d,...>
    parts = []
    for (a, b) in intervals:
        if a == b:
            parts.append(str(a))
        else:
            parts.append(f"{a}-{b}")
    return "UNRECOVERABLE:" + ",".join(parts)


def _verify(payload, placed, L, reassemblable, answer):
    """Check gold answer reproduces the target state."""
    if answer == payload:
        return reassemblable and len(answer) == L
    if answer.startswith("UNRECOVERABLE:"):
        cov = _coverage(placed, L)
        expect = _answer_string(payload, placed, L, _can_fully_reassemble(placed, L))
        return answer == expect and not reassemblable
    return False


def _prompt(frags, missing_ids, overlap_ms, L):
    lines = []
    lines.append("A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.")
    lines.append("- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.")
    lines.append("- The payload has a known total length of %d characters (positions 0 .. %d)." % (L, L - 1))
    lines.append(f"- Overlap tolerance between adjacent fragments is at most {overlap_ms} characters.")
    lines.append("")
    lines.append("Fragments:")
    for f in frags:
        if f.missing:
            lines.append("  - index %d, offset %d: MISSING" % (f.index, f.offset))
        else:
            lines.append("  - index %d, offset %d: %s" % (f.index, f.offset, f.data))
    lines.append("")
    lines.append("Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.")
    return "\n".join(lines)


class PacketFragmentReassembly(Task):
    summary = "Reassemble indexed or offset message fragments under explicit overlap and missing-fragment rules, returning the reconstructed payload or error state."
    config_cls = PacketFragmentReassemblyConfig

    def generate_entry(self):
        cfg = self.config
        L = random.randint(int(cfg.min_payload), int(cfg.max_payload))
        payload = _build_payload(L, random)
        n_fragments = int(cfg.n_fragments)
        frags, placed = _place_fragments(payload, n_fragments, int(cfg.overlap_ms), random)
        reassemblable = _can_fully_reassemble(placed, L)
        # ensure the answer domain is respected: payload length fixed, error string valid
        answer = _answer_string(payload, placed, L, reassemblable)
        # domain sanity table
        assert isinstance(L, int) and L > 0
        assert isinstance(answer, str) and len(answer) > 0
        # a count-like or bounded quantity domain:
        #   reassemblable implies answer is the exact payload of length L
        #   error implies the interval list covers exactly the uncovered positions
        if not _verify(payload, placed, L, reassemblable, answer):
            raise RuntimeError("gold answer verifier rejected instance")
        gaps = [i for i in range(L) if not _coverage(placed, L)[i]]
        metadata = edict({
            "payload": payload,
            "length": L,
            "fragments": [{"index": f.index, "offset": f.offset, "data": f.data, "missing": f.missing} for f in frags],
            "overlap_ms": int(cfg.overlap_ms),
            "n_fragments": n_fragments,
            "reassemblable": reassemblable,
            "gap_positions": gaps,
        })
        metadata.payload_prompt = _prompt(frags, [f.index for f in frags if f.missing], int(cfg.overlap_ms), L)
        # JSON-serializable: metadata.payload holds only plain values
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return metadata.payload_prompt

    def score_answer(self, answer, entry):
        # answer and entry here: no self access allowed
        gold = _parse_answer(entry.answer)
        if gold is None:
            return 0.0
        parsed = _parse_answer(answer)
        if parsed is None:
            return 0.0
        return 1.0 if parsed == gold else 0.0


def _parse_answer(text):
    """Parse a candidate answer string into canonical form for comparison."""
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    # payload form: must be all alphabet chars
    if re.fullmatch(r"[A-H]+", t):
        return ("payload", t)
    if t.startswith("UNRECOVERABLE:"):
        body = t[len("UNRECOVERABLE:"):].strip()
        if body == "":
            return None
        return ("error", body)
    return None
