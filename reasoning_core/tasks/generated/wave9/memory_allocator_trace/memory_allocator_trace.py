import random
from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


class MemoryAllocatorConfig(Config):
    n_ops: int = 4
    n_blocks: int = 3
    max_size: int = 8
    strategy: str = "first_fit"

    def apply_difficulty(self, level):
        self.n_ops = sround(self.n_ops + level)
        self.n_blocks = sround(self.n_blocks + (level // 2))
        self.max_size = sround(self.max_size + level)


def _layout_str(spans):
    return " ".join(f"[{l},{h})" for l, h in spans)


def _simulate(blocks, cmd, strategy):
    # Each block: sorted list of (lo, hi, 'free'|'reserved') segments covering the block.
    mem = []
    for blk in blocks:
        base, size = blk
        mem.append([(base, base + size, "free")])

    addr_next = 0
    allocs = {}

    for op, arg in cmd:
        if op == "alloc":
            size = arg
            chosen = None
            if strategy == "first_fit":
                for i, seg in enumerate(mem):
                    if sum(h - l for l, h, s in seg if s == "free") >= size:
                        chosen = i
                        break
            else:
                best = None
                for i, seg in enumerate(mem):
                    fs = sum(h - l for l, h, s in seg if s == "free")
                    if fs >= size and (best is None or fs < best[0]):
                        best = (fs, i)
                chosen = best[1] if best else None
            if chosen is not None:
                seg = mem[chosen]
                for j, (l, h, s) in enumerate(seg):
                    if s == "free" and h - l >= size:
                        addr = addr_next
                        addr_next += 1
                        seg[j:j + 1] = [(l, l + size, "reserved"), (l + size, h, "free")]
                        allocs[addr] = (chosen, (l, l + size))
                        break
        else:
            addr = arg
            if addr not in allocs:
                continue
            chosen, (l0, h0) = allocs.pop(addr)
            seg = mem[chosen]
            for j, (l, h, s) in enumerate(seg):
                if s == "reserved" and (l, h) == (l0, h0):
                    seg[j] = (l, h, "free")
                    break
            merged = []
            for l, h, s in seg:
                if merged and merged[-1][2] == "free" and s == "free" and merged[-1][1] == l:
                    merged[-1] = (merged[-1][0], h, "free")
                else:
                    merged.append((l, h, s))
            mem[chosen] = merged

    spans = []
    for seg in mem:
        for l, h, s in seg:
            if s == "free" and h > l:
                spans.append([l, h])
    return spans


class MemoryAllocatorTrace(Task):
    summary = ("Simulate first-fit or best-fit allocation, freeing, splitting, "
               "and adjacent-block coalescing, returning allocation addresses "
               "or final block layout.")
    config_cls = MemoryAllocatorConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        strategy = cfg.strategy
        n_blocks = cfg.n_blocks
        n_ops = cfg.n_ops
        max_size = cfg.max_size

        while True:
            blocks = []
            for _ in range(n_blocks):
                size = random.randint(3, max(3, max_size + 1))
                blocks.append([0, size])

            # Build a valid trace with a live simulation so every allocated
            # op succeeds; freed addresses are drawn from currently live ones.
            mem = []
            for blk in blocks:
                mem.append([(blk[0], blk[1], "free")])
            cmd = []
            live = {}  # addr -> (block_idx, (lo, hi)) span
            addr_next = 0
            opcount = 0
            while opcount < n_ops:
                if live and random.random() < 0.4:
                    addr = random.choice(sorted(live.keys()))
                    chosen, (l0, h0) = live.pop(addr)
                    cmd.append(("free", addr))
                    seg = mem[chosen]
                    for j, (l, h, s) in enumerate(seg):
                        if s == "reserved" and (l, h) == (l0, h0):
                            seg[j] = (l, h, "free")
                            break
                    merged = []
                    for l, h, s in seg:
                        if merged and merged[-1][2] == "free" and s == "free" and merged[-1][1] == l:
                            merged[-1] = (merged[-1][0], h, "free")
                        else:
                            merged.append((l, h, s))
                    mem[chosen] = merged
                    opcount += 1
                else:
                    size = random.randint(1, max(1, max_size))
                    chosen_block = None
                    if strategy == "first_fit":
                        for i, seg in enumerate(mem):
                            if sum(h - l for l, h, s in seg if s == "free") >= size:
                                chosen_block = i
                                break
                    else:
                        best = None
                        for i, seg in enumerate(mem):
                            fs = sum(h - l for l, h, s in seg if s == "free")
                            if fs >= size and (best is None or fs < best[0]):
                                best = (fs, i)
                        chosen_block = best[1] if best else None
                    if chosen_block is None:
                        continue  # no block fits; retry a different op
                    seg = mem[chosen_block]
                    for j, (l, h, s) in enumerate(seg):
                        if s == "free" and h - l >= size:
                            addr = addr_next
                            addr_next += 1
                            seg[j:j + 1] = [(l, l + size, "reserved"), (l + size, h, "free")]
                            live[addr] = (chosen_block, (l, l + size))
                            cmd.append(("alloc", size))
                            break
                    opcount += 1

            # Simulate from scratch to produce the gold answer.
            spans = _simulate(blocks, cmd, strategy)
            layout = _layout_str(spans)
            if not spans:
                continue
            re = _simulate(blocks, cmd, strategy)
            if _layout_str(re) != layout:
                continue

            meta = edict({
                "strategy": strategy,
                "blocks": blocks,
                "ops": [list(o) if isinstance(o, tuple) else o for o in cmd],
                "layout": spans,
            })
            meta.payload = {
                "strategy": strategy,
                "blocks": blocks,
                "ops": [list(o) if isinstance(o, tuple) else o for o in cmd],
            }
            return Entry(metadata=meta, answer=layout)

    def render_prompt(self, metadata):
        payload = {
            "strategy": metadata.strategy,
            "blocks": metadata.blocks,
            "ops": metadata.ops,
        }
        return (render_payload(payload)
                + "\n\nThe answer is the final free-memory layout: each free span in "
                  "[lo,hi) form, block by block in the given order, joined by spaces.")

    def score_answer(self, answer, entry):
        try:
            return 1.0 if str(answer).strip() == entry.answer else 0.0
        except Exception:
            return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'memory_allocator_trace (draw 1 of 1)',
 'hypothesis': 'HV-042',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/memory_allocator_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 103415150,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
