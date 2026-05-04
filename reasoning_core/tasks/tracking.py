import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from z3 import Int, Solver, sat

from reasoning_core.template import Task, Problem, Config, edict


# Symmetric inverse of a size relation, used for deduplication.
_INV_REL = {">": "<", "<": ">", "==": "=="}


def _qualify_desc(d: str) -> str:
    """'the ball in xK' -> 'the ball that started in xK' (avoids leaking final box)."""
    return re.sub(r"^the ball in (x\d+)$", r"the ball that started in \1", d)


@dataclass
class ReferenceTrackingConfig(Config):
    n_balls: int = 4
    n_boxes: int = 3
    n_steps: int = 4
    bulk_move_p: float = 0.25
    pronoun_move_p: float = 0.3

    prefer_indirect_p: float = 0.2  # Bumped to force tracking coupling in Winograd zero-level
    winograd_p: float = 0.35
    chain_len: int = 2
    extra_facts: int = 2
    allow_equalities_p: float = 0.5
    ask_location_p: float = 0.5
    winograd_singles_p: float = 0.5

    def update(self, c=1):
        self.n_balls     += 0.6 * c
        self.n_boxes     += 0.3 * c
        self.n_steps     += 0.7 * c
        self.chain_len   += 0.4 * c
        self.extra_facts += 0.5 * c

        self.bulk_move_p        = min(0.75, self.bulk_move_p + 0.03 * c)
        self.pronoun_move_p     = min(0.55, self.pronoun_move_p + 0.04 * c)
        self.prefer_indirect_p  = min(1.0,  self.prefer_indirect_p + 0.2 * c)
        self.winograd_p         = min(0.85, self.winograd_p + 0.04 * c)
        self.allow_equalities_p = min(0.9,  self.allow_equalities_p + 0.08 * c)
        self.ask_location_p     = min(0.9,  self.ask_location_p + 0.08 * c)
        self.winograd_singles_p = min(0.9,  self.winograd_singles_p + 0.1 * c)


class ReferenceTracking(Task):
    """
    Two interleaved skills tested together:

    1. State tracking  — follow individual, pronoun, and bulk ball moves across boxes.
    2. Coreference     — resolve "it" in a logical-Winograd docking failure
                         using Z3-verified size-chain facts, then look up
                         the referent's current box (forcing both skills).
    """

    def __init__(self, config=ReferenceTrackingConfig()):
        super().__init__(config=config)



    def _box_inv(
        self, placement: Dict[str, str], boxes: List[str], balls: List[str]
    ) -> Dict[str, List[str]]:
        inv: Dict[str, List[str]] = {x: [] for x in boxes}
        for b in balls:
            inv[placement[b]].append(b)
        return inv

    def _single_occupants(
        self, placement: Dict[str, str], boxes: List[str], balls: List[str]
    ) -> Dict[str, str]:
        """Return {box: ball} for boxes holding exactly one ball."""
        return {
            box: bs[0]
            for box, bs in self._box_inv(placement, boxes, balls).items()
            if len(bs) == 1
        }

    def _pick_desc(
        self,
        target: str,
        balls: List[str],
        colors: Dict[str, str],
        placement: Dict[str, str],
        boxes: List[str],
        prefer_indirect: bool,
    ) -> str:
        if not prefer_indirect:
            return target
        singles = self._single_occupants(placement, boxes, balls)
        for box, ball in singles.items():
            if ball == target:
                return f"the ball in {box}"
        col = colors[target]
        if sum(1 for b in balls if colors[b] == col) == 1:
            return f"the {col} ball"
        return target

    def _do_moves(
        self,
        placement: Dict[str, str],
        balls: List[str],
        boxes: List[str],
        n_steps: int,
        bulk_p: float,
        pronoun_p: float,
    ) -> Tuple[List[str], List[str]]:
        sents: List[str] = []
        resolved: List[str] = []
        last_explicit: Optional[str] = None

        for _ in range(n_steps):
            if len(boxes) >= 2 and random.random() < bulk_p:
                inv = self._box_inv(placement, boxes, balls)
                nonempty = [x for x, bs in inv.items() if bs]
                if nonempty:
                    src = random.choice(nonempty)
                    dst = random.choice([x for x in boxes if x != src])
                    for b in balls:
                        if placement[b] == src:
                            placement[b] = dst
                    t = random.choice([
                        f"Move all contents of {src} to {dst}.",
                        f"Transfer everything in {src} into {dst}.",
                        f"Relocate all balls from {src} to {dst}."
                    ])
                    sents.append(t)
                    resolved.append(f"Move all contents of {src} to {dst}.")
                    last_explicit = None
                continue

            # Pronoun reuse: "Move it ..." for the last explicitly named ball.
            if last_explicit is not None and random.random() < pronoun_p:
                b = last_explicit
                src = placement[b]
                candidates = [x for x in boxes if x != src]
                if not candidates:
                    continue
                dst = random.choice(candidates)
                placement[b] = dst
                sents.append(f"Move it from {src} to {dst}.")
                resolved.append(f"Move {b} from {src} to {dst}.")
                last_explicit = None
            else:
                b = random.choice(balls)
                src = placement[b]
                candidates = [x for x in boxes if x != src]
                if not candidates:
                    continue
                dst = random.choice(candidates)
                placement[b] = dst
                # Avoiding "it" in standard phrases to prevent coreference confusion
                t = random.choice([
                    f"Move {b} from {src} to {dst}.",
                    f"Transfer {b} from {src} into {dst}.",
                    f"Relocate {b} from {src} to {dst}."
                ])
                sents.append(t)
                resolved.append(f"Move {b} from {src} to {dst}.")
                last_explicit = b

        return sents, resolved

    @staticmethod
    def _rel_text(a: str, rel: str, b: str) -> str:
        if rel == ">":  return f"{a} is larger than {b}."
        if rel == "<":  return f"{a} is smaller than {b}."
        return f"{a} is the same size as {b}."

    def _build_size_facts(
        self,
        balls: List[str],
        larger: str,
        smaller: str,
        chain_len: int,
        extra_facts: int,
        allow_eq: bool,
    ) -> List[Tuple[str, str, str]]:
        sz = {x: Int(f"sz_{x}") for x in balls}
        max_val = max(30, 10 * len(balls))
        constraints = [sz[x] >= 1 for x in balls] + [sz[x] <= max_val for x in balls]

        mids = [b for b in balls if b not in {larger, smaller}]
        random.shuffle(mids)
        chain = [larger] + mids[: max(0, chain_len - 1)] + [smaller]

        facts: List[Tuple[str, str, str]] = []
        seen: set = set()

        for i in range(len(chain) - 1):
            x, y = chain[i], chain[i + 1]
            rel = ">" if (i == 0 or not allow_eq) else random.choice([">", "=="])
            facts.append((x, rel, y))
            seen.update({(x, rel, y), (y, _INV_REL[rel], x)})
            constraints.append(sz[x] > sz[y] if rel == ">" else sz[x] == sz[y])

        pairs = [(u, v) for u in balls for v in balls if u != v]
        random.shuffle(pairs)
        added = 0
        for u, v in pairs:
            if added >= extra_facts:
                break
            rel = random.choice([">", "<"] + (["=="] if allow_eq else []))
            if (u, rel, v) in seen:
                continue
            c_expr = (
                sz[u] > sz[v] if rel == ">" else
                sz[u] < sz[v] if rel == "<" else
                sz[u] == sz[v]
            )
            tmp = Solver()
            tmp.add(constraints + [c_expr])
            if tmp.check() == sat:
                constraints.append(c_expr)
                facts.append((u, rel, v))
                seen.update({(u, rel, v), (v, _INV_REL[rel], u)})
                added += 1

        return facts

    def generate(self) -> Problem:
        c = self.config

        chain_len = max(1, int(c.chain_len))
        # Enforce enough balls to always support the requested strict chain length
        n_balls = max(3, int(c.n_balls), chain_len + 1)
        n_boxes = max(2, int(c.n_boxes))
        balls   = [f"b{i+1}" for i in range(n_balls)]
        boxes   = [f"x{i+1}" for i in range(n_boxes)]

        colors = {
            b: random.choice(["red", "blue", "green", "yellow", "black", "white"])
            for b in balls
        }
        initial_placement = {b: random.choice(boxes) for b in balls}
        placement = dict(initial_placement)

        moves, resolved_moves = self._do_moves(
            placement, balls, boxes,
            int(c.n_steps), float(c.bulk_move_p), float(c.pronoun_move_p),
        )
        prefer_indirect = random.random() < float(c.prefer_indirect_p)

        # ---- pure tracking ----
        if random.random() >= float(c.winograd_p):
            target = random.choice(balls)
            desc = _qualify_desc(
                self._pick_desc(target, balls, colors, initial_placement, boxes, prefer_indirect)
            )
            return Problem(
                metadata=edict(
                    family="track", balls=balls, boxes=boxes, colors=colors,
                    initial_placement=initial_placement,
                    moves=moves, resolved_moves=resolved_moves, 
                    final_placement=dict(placement),
                    question=f"Where is {desc} now? The answer is a box tag, like x1.",
                ),
                answer=placement[target],
            )

        # ---- logical winograd ----
        ask_loc = random.random() < float(c.ask_location_p)
        desc_snapshot = initial_placement if ask_loc else placement

        singles = list(self._single_occupants(desc_snapshot, boxes, balls).values())
        if len(singles) >= 2 and random.random() < float(c.winograd_singles_p):
            a, b = random.sample(singles, 2)
        else:
            a, b = random.sample(balls, 2)

        larger  = random.choice([a, b])
        smaller = b if larger == a else a

        facts = self._build_size_facts(
            balls=balls, larger=larger, smaller=smaller,
            chain_len=chain_len,
            extra_facts=int(c.extra_facts),
            allow_eq=random.random() < float(c.allow_equalities_p),
        )
        reason   = random.choice(["too large", "too small"])
        referent = larger if reason == "too large" else smaller

        desc_a = self._pick_desc(a, balls, colors, desc_snapshot, boxes, prefer_indirect)
        desc_b = self._pick_desc(b, balls, colors, desc_snapshot, boxes, prefer_indirect)
        
        if ask_loc:
            desc_a = _qualify_desc(desc_a)
            desc_b = _qualify_desc(desc_b)

        if ask_loc:
            question = (
                "Question: Which box is the thing referred to by 'it' "
                "in the last sentence currently in? Answer with a box tag like x1."
            )
            answer = placement[referent]
        else:
            question = (
                "Question: What does 'it' refer to in the last sentence? "
                "Answer with a ball tag like b1."
            )
            answer = referent

        return Problem(
            metadata=edict(
                family="logical_winograd",
                balls=balls, boxes=boxes, colors=colors,
                initial_placement=initial_placement,
                moves=moves, resolved_moves=resolved_moves,
                final_placement=dict(placement),
                size_facts=facts,
                size_facts_text=[self._rel_text(u, r, v) for u, r, v in facts],
                dock=dict(
                    a=a, b=b, desc_a=desc_a, desc_b=desc_b,
                    reason=reason, larger=larger, smaller=smaller,
                ),
                gold_referent=referent,
                question=question,
            ),
            answer=answer,
        )

    def prompt(self, metadata) -> str:
        m = metadata
        inv  = "Inventory:\n"     + "\n".join(f"- {b}: {m.colors[b]}" for b in m.balls)
        init = "Initial state:\n" + "\n".join(f"- {b} is in {m.initial_placement[b]}" for b in m.balls)
        mvs  = "Moves:\n"         + ("\n".join(f"- {s}" for s in m.moves) if m.moves else "- (none)")

        if m.family == "track":
            return "\n".join([inv, init, mvs, m.question])

        rules = (
            "Rules:\n"
            "- Each ball has a positive integer size.\n"
            "- Dock(X, Y) succeeds iff size(X) == size(Y).\n"
            "- If docking fails and the failure sentence says 'it was too large/small',\n"
            "  'it' refers to the larger/smaller of the two docked balls.\n"
        )
        facts = "Size facts:\n" + "\n".join(f"- {t}" for t in m.size_facts_text)
        story = (
            "Story:\n"
            f"- After the moves, Rae tried to dock {m.dock.desc_a} with {m.dock.desc_b}.\n"
            f"- It failed because it was {m.dock.reason}."
        )
        return "\n".join([rules, inv, init, mvs, facts, story, m.question])

    def score_answer(self, answer: str, entry: Problem) -> float:

        norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

        gold = norm(entry.answer)
        ans  = str(answer).lower()
        if norm(ans) == gold:
            return 1.0
        
        prefix = gold[0] 
        found  = set(re.findall(rf"{prefix}\d+", ans))
        if len(found) == 1 and norm(next(iter(found))) == gold:
            return 1.0
        return 0.0