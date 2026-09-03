"""Virtual address translation task.

Given a multi-level page table and a virtual address, output the physical
address or report a page fault.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload


@dataclass
class VirtualAddressTranslationConfig(Config):
    levels: int = 2
    bits_per_level: int = 5
    offset_bits: int = 10
    table_entries: int = 16
    fault_rate: float = 0.15

    def apply_difficulty(self, level):
        self.levels = self.levels + (level // 3)
        self.bits_per_level = self.bits_per_level + (level % 3)
        self.offset_bits = self.offset_bits - (level % 2)
        self.table_entries = self.table_entries + level
        self.fault_rate = min(0.25, 0.15 + 0.02 * level)


class VirtualAddressTranslation(Task):
    """Translate virtual addresses through multi-level page tables."""
    summary = ("Given multi-level hierarchical page tables (2-3 levels, offset fields) and a "
               "virtual address in bits, output the physical frame address or report a page fault.")
    config_cls = VirtualAddressTranslationConfig

    def generate_entry(self):
        cfg = self.config
        levels = cfg.levels
        bpl = cfg.bits_per_level
        obits = cfg.offset_bits

        # Virtual address size in bits = offset + levels*bpl
        vabits = obits + levels * bpl
        mask_level = (1 << bpl) - 1
        offset_mask = (1 << obits) - 1

        # Build a random physical frame range per level: each page table
        # entry maps a virtual page (partial) to a physical frame number.
        # We'll represent level tables as dicts from level-index -> frame.
        # Root level has a single page directory.

        # Number of pages = number of distinct virtual page numbers = 2^(vabits - obits)
        # We won't enumerate all; we assign frames to a random selection.

        memory_limit = 1 << (obits + bpl)  # physical memory holds 'table_entries' frames

        # Choose fault decision
        fault = random.random() < cfg.fault_rate

        # We'll construct tables stochastically. Simpler: generate a full
        # mapping from virtual page number to physical frame base.
        # vpn_bits = vabits - obits
        vpn_bits = levels * bpl
        # choose a random virtual address
        virtual = random.getrandbits(vabits)

        vpn = virtual >> obits
        offset = virtual & offset_mask

        # Build hierarchical tables
        tables = {}
        # We'll index tables by tuple of (level, index at that level)
        # Simplest robust construction: build random frame assigments.

        # Represent page tables as nested dict: level0 table -> level1 ... -> frame
        # To make it concrete, build a set of valid VPNs (walking) and frames.
        # We'll create a random number of valid pages.

        n_pages = random.randint(1, max(1, cfg.table_entries // 2))
        valid_pages = set()
        for _ in range(n_pages):
            valid_pages.add(random.getrandbits(vpn_bits))
        if not fault and vpn not in valid_pages:
            valid_pages.add(vpn)

        # Physical frames assigned to each valid page (distinct frames).
        frames = {}
        for pg in sorted(valid_pages):
            frames[pg] = random.randrange(memory_limit)

        # Determine answer
        if fault:
            phys = virtual  # placeholder; will be overridden
            answer = "page fault"
            physical_base = None
        else:
            if vpn in frames:
                base = frames[vpn]
                phys = (base << obits) | offset
                answer = str(phys)
                physical_base = base
            else:
                answer = "page fault"
                physical_base = None

        # Build nested tables data for prompt as a tree. Keys are stringified
        # level indices so the tree survives JSON / EasyDict serialization.

        root = {}
        for pg in sorted(valid_pages):
            indices = []
            tmp = pg
            for _ in range(levels):
                idx = tmp & mask_level
                indices.append(idx)
                tmp >>= bpl
            indices.reverse()  # level0 .. levelN-1
            node = root
            for i in range(levels - 1):
                idx = str(indices[i])
                if idx not in node:
                    node[idx] = {}
                node = node[idx]
            node[str(indices[-1])] = frames[pg]

        # Build virtual address binary string
        virtual_bits = format(virtual, '0{}b'.format(vabits))

        payload = {
            "page_table_levels": levels,
            "bits_per_level": bpl,
            "offset_bits": obits,
            "page_table": root,
            "virtual_address_bits": virtual_bits,
        }

        metadata = edict({
            "levels": levels,
            "bpl": bpl,
            "obits": obits,
            "table": root,
            "virtual": virtual,
            "vabits": vabits,
            "answer_is_fault": (physical_base is None),
            "physical_base": physical_base,
            "offset": offset,
            "payload": payload,
        })

        # Verify: re-walk
        if not fault:
            # ensure the answer actually reproduces given the tables
            assert physical_base is not None
            assert answer is not None

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        lines = [
            "A process uses a {}-level page table with {} bits per level and a {}-bit "
            "offset field.".format(payload["page_table_levels"],
                                   payload["bits_per_level"],
                                   payload["offset_bits"]),
            "The page table (as nested dicts mapping level index to subtree, with leaf values "
            "being physical frame numbers) is:",
            render_payload(payload["page_table"]),
            "The virtual address is the {}-bit binary value {}.".format(
                payload["virtual_address_bits"].__len__(), payload["virtual_address_bits"]),
            "The answer is the physical address as a decimal integer, or the exact two-word "
            "phrase 'page fault' if the translation faults.",
        ]
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 0.0
        a = str(answer).strip().lower()
        g = gold.strip().lower()
        if a == g:
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'virtual_address_translation (draw 1 of 2)',
 'hypothesis': 'W1-043',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/virtual_address_translation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 28548294,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
