import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'buddy_allocator (draw 1 of 2)',
 'hypothesis': 'W1-049',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/buddy_allocator',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 545695329,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _order_for(n):
    o = 0
    while (1 << o) < n:
        o += 1
    return o


class Buddy:
    """True power-of-two buddy allocator keyed by block start address."""

    def __init__(self, order):
        self.order = order
        self.total = 1 << order
        self.free = {}   # addr -> size
        self.alloc = {}  # addr -> size
        half = self.total // 2
        self.free[0] = half
        self.free[half] = self.total - half

    def buddy(self, addr, size):
        return addr ^ size

    def split(self, addr, size):
        del self.free[addr]
        half = size // 2
        self.free[addr] = half
        self.free[addr + half] = half

    def allocate(self, size):
        cur = size
        while cur <= self.total:
            addrs = [a for a in self.free if self.free[a] == cur]
            if addrs:
                a = random.choice(addrs)
                while self.free[a] > size:
                    adult = self.free[a]
                    self.split(a, adult)
                del self.free[a]
                self.alloc[a] = size
                return a
            cur *= 2
        return None

    def merge_free(self, addr, size):
        while size < self.total:
            b = self.buddy(addr, size)
            if self.free.get(b) == size:
                del self.free[addr]
                del self.free[b]
                addr = min(addr, b)
                size *= 2
                self.free[addr] = size
            else:
                break

    def free_block(self, addr):
        size = self.alloc.pop(addr)
        self.free[addr] = size
        self.merge_free(addr, size)

    def free_sizes(self):
        sizes = sorted(self.free.values(), reverse=True)
        return sizes

    def free_pairs(self):
        return sorted((a, self.free[a]) for a in self.free)


@dataclass
class BuddyConfig(Config):
    order_cap: int = 7
    setup_ops: int = 6

    def apply_difficulty(self, level):
        self.order_cap = sround(self.order_cap + level)
        self.setup_ops = sround(self.setup_ops + level)


def _format_sizes(sizes):
    if not sizes:
        return "empty"
    counts = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    parts = []
    for s in sorted(counts.keys(), reverse=True):
        parts.append(f"{counts[s]}x{s}")
    return "; ".join(parts)


def _free_pairs_str(pairs):
    if not pairs:
        return "none"
    return ", ".join(f"{size}@start{addr}" for addr, size in pairs)


class BuddyAllocator(Task):
    summary = ("Apply one allocate or one free to a power-of-two buddy allocator (address-pinned "
               "free list given) and output the canonical post-operation free-size multiset "
               "(count per size, sizes desc, or 'empty'), across orders 1-9 with random splits, "
               "frees and buddy merges.")
    config_cls = BuddyConfig

    def generate_entry(self):
        max_order = random.randrange(4, 9)
        order = random.randrange(1, max_order + 1)
        b = Buddy(order)
        total = b.total

        setup = random.randrange(2, 9)
        for _ in range(setup):
            if random.random() < 0.6:
                sizes = b.free_sizes()
                if sizes:
                    req = random.choice(sizes)
                    if random.random() < 0.5:
                        req = max(1, req // 2)
                    b.allocate(req)
            else:
                if b.alloc:
                    addr = random.choice(sorted(b.alloc))
                    b.free_block(addr)

        can_free = bool(b.alloc)
        can_alloc = bool(b.free)
        if can_free and can_alloc:
            is_free_op = random.random() < 0.5
        else:
            is_free_op = can_free and not can_alloc

        before_pairs = b.free_pairs()
        if is_free_op:
            op = "free"
            addr = random.choice(sorted(b.alloc))
            size = b.alloc[addr]
            b.free_block(addr)
        else:
            op = "allocate"
            sizes = sorted(b.free_sizes())
            if len(sizes) == 1:
                big = sizes[0]
                half = big // 2
                if half >= 1:
                    req = random.randrange(1, half + 1)
                    size = 1 << _order_for(req)
                    while size > half:
                        size //= 2
                    if size < 1:
                        size = 1
                else:
                    size = 1
            else:
                size = random.choice(sizes)
            b.allocate(size)

        after_sizes = b.free_sizes()
        answer = _format_sizes(after_sizes)
        total_free = sum(after_sizes) if after_sizes else 0
        if total_free != (b.total - sum(b.alloc.values())):
            raise RuntimeError("free total mismatch")

        metadata = edict({
            "order": order,
            "op": op,
            "before_pairs": before_pairs,
            "op_addr": addr if op == "free" else None,
            "op_size": size,
            "after_sizes": after_sizes,
            "payload": {
                "order": order,
                "op": op,
                "free_before": _free_pairs_str(before_pairs),
                "op_addr": addr if op == "free" else None,
                "op_size": size,
            },
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        if metadata.op == "allocate":
            opdesc = f"Allocate a block of size {metadata.op_size}."
        else:
            opdesc = (f"Free the currently-allocated block of size {metadata.op_size} "
                      f"starting at {metadata.op_addr}.")
        return (
            f"This is a power-of-two buddy allocator of maximum order {metadata.order}; every "
            f"block size is a power of two up to 2^{metadata.order}, and a block's buddy is its "
            f"sibling from the split that created it. When a block is freed, it merges with its "
            f"buddy whenever the buddy is free, repeatedly. Currently the free blocks are: "
            f"{metadata.payload['free_before']} (all other blocks are occupied).\n\n"
            f"{opdesc} Afterwards, report the free blocks as a multiset of sizes: for each "
            f"distinct free size list 'count x size', sizes in decreasing order, separated by "
            f"'; ', or write 'empty' if none remain. The answer is that list."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        text = answer.strip()
        if text == "empty":
            return 1.0 if entry.answer.strip() == "empty" else 0.0
        if entry.answer.strip() == "empty":
            return 0.0
        try:
            got = _parse_multiset(text)
            exp = _parse_multiset(entry.answer)
        except Exception:
            return 0.0
        return 1.0 if got == exp else 0.0


def _parse_multiset(text):
    parts = [p.strip() for p in text.split(";") if p.strip()]
    result = []
    for part in parts:
        if "x" not in part:
            raise ValueError("bad format")
        c, s = part.split("x", 1)
        result.append((int(c), int(s)))
    return sorted(result, reverse=True)
