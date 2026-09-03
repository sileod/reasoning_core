import random
import re

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'firewall_rule_shadowing (draw 1 of 2)',
 'hypothesis': 'W1-052',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/firewall_rule_shadowing',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2546520183,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _parse_rule(tokens):
    proto, src_lo, src_hi = tokens[0], int(tokens[1]), int(tokens[2])
    dst_lo, dst_hi = int(tokens[3]), int(tokens[4])
    action = tokens[5]
    return (proto, src_lo, src_hi, dst_lo, dst_hi, action)


def _intersects(p1, s1_lo, s1_hi, d1_lo, d1_hi, p2, s2_lo, s2_hi, d2_lo, d2_hi):
    if p1 != p2:
        return False
    return (s1_lo <= s2_hi and s2_lo <= s1_hi) and (d1_lo <= d2_hi and d2_lo <= d1_hi)


def _contains(p1, s1_lo, s1_hi, d1_lo, d1_hi, p2, s2_lo, s2_hi, d2_lo, d2_hi):
    if p1 != p2:
        return False
    return (s1_lo <= s2_lo and s2_hi <= s1_hi) and (d1_lo <= d2_lo and d2_hi <= d1_hi)


class FirewallRuleShadowingConfig(Config):
    n_rules: int = 6
    addr_max: int = 128

    def apply_difficulty(self, level):
        self.n_rules = 6 + level * 3
        self.addr_max = 64 + level * 96


class FirewallRuleShadowing(Task):
    summary = ("Given ordered finite-domain firewall rules over protocol/src/dst "
               "ranges, output the first fully shadowed rule index/action/range or None.")

    config_cls = FirewallRuleShadowingConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_rules)
        amax = int(cfg.addr_max)
        protos = ["tcp", "udp"]
        actions = ["allow", "deny"]

        while True:
            rules = []
            for _ in range(n):
                proto = random.choice(protos)
                lo1 = random.randint(0, amax - 1)
                hi1 = random.randint(lo1, amax - 1)
                lo2 = random.randint(0, amax - 1)
                hi2 = random.randint(lo2, amax - 1)
                action = random.choice(actions)
                rules.append([proto, lo1, hi1, lo2, hi2, action])

            shadow_states = []
            for i in range(n):
                pi, slo, shi, dlo, dhi, a = rules[i]
                shadowed = False
                covered = False
                for j in range(i):
                    pj, jlo, jhi, klo, khi, b = rules[j]
                    if _contains(pj, jlo, jhi, klo, khi, pi, slo, shi, dlo, dhi):
                        covered = True
                        if b == a:
                            shadowed = True
                            break
                shadow_states.append(("shadowed" if shadowed else "covered" if covered else "own"))

            first_shadow = None
            for i, st in enumerate(shadow_states):
                if st == "shadowed":
                    first_shadow = i
                    break

            if first_shadow is None:
                continue
            if not any(st == "own" for st in shadow_states):
                continue

            pi, slo, shi, dlo, dhi, a = rules[first_shadow]
            answer = "%d %s %d-%d %d-%d %s" % (first_shadow, pi, slo, shi, dlo, dhi, a)

            # verify: first fully shadowed rule
            found = None
            for i in range(n):
                p2, s2_lo, s2_hi, d2_lo, d2_hi, a2 = rules[i]
                is_shadow = False
                for j in range(i):
                    pj, jlo, jhi, klo, khi, b = rules[j]
                    if _contains(pj, jlo, jhi, klo, khi, p2, s2_lo, s2_hi, d2_lo, d2_hi) and b == a2:
                        is_shadow = True
                        break
                if is_shadow:
                    found = i
                    break
            if found is None or found != first_shadow:
                continue
            break

        payload_rules = ["%d %s %d-%d %d-%d %s" % (i, *rules[i]) for i in range(n)]
        metadata = edict({})
        metadata.rules = payload_rules
        metadata.n_rules = int(n)
        metadata.payload = {"rules": payload_rules, "n_rules": int(n)}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = "\n".join(metadata.payload["rules"])
        return (
            "An ordered firewall processes packets top to bottom: the first matching "
            "rule decides the action (allow or deny), later rules are ignored for that "
            "packet. A rule is fully shadowed when some strictly earlier rule matches "
            "every packet that it would match and has the same action, so the later rule "
            "can never affect any packet.\n"
            "Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, "
            "where the protocol is tcp or udp and SRC/DST are inclusive address ranges "
            "from 0 upward. Two rules match the same packet only if their protocols "
            "are equal and both the source ranges and the destination ranges overlap.\n"
            "Rules:\n%s\n\n"
            "Which is the FIRST rule (smallest index) that is fully shadowed by an "
            "earlier rule? Output its rule verbatim in the exact format "
            "INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no "
            "rule is fully shadowed.\nThe answer is the rule itself."
        ) % lines

    def score_answer(self, answer, entry):
        if isinstance(answer, str):
            cand = re.sub(r"\s+", " ", answer.strip()).strip()
        else:
            cand = ""
        if cand == "":
            return 0.0
        gold = entry.answer.strip()
        gold_norm = re.sub(r"\s+", " ", gold).strip()
        if cand.lower() in ("none", "none."):
            return 1.0 if gold_norm == "None" else 0.0
        if cand == gold_norm:
            return 1.0
        return 0.0
