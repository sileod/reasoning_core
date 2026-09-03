import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'raft_vote_eligibility (draw 1 of 2)',
 'hypothesis': 'W1-042',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/raft_vote_eligibility',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2471421591,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

MAX_TERM = 6
MAX_INDEX = 25


def _decide(Tc, Lc, Ic, Vv, Lv, Iv):
    """Return (mode, witness) given candidate term Tc, candidate last log (Lc,Ic),
    voter last voted for term Vv, voter last log (Lv,Iv)."""
    if Tc <= Vv:
        return "stale", Vv
    if Lc > Lv or (Lc == Lv and Ic >= Iv):
        return "grant", Tc
    return "log", Iv


@dataclass
class RaftVoteEligibilityConfig(Config):
    n_candidates: int = 4
    max_term: int = 6
    max_index: int = 25

    def apply_difficulty(self, level):
        self.n_candidates = int(3 + level)
        self.max_term = int(4 + level)
        self.max_index = int(15 + 5 * level)


class RaftVoteEligibility(Task):
    summary = ("Given candidate and voter terms plus log metadata, answer whether the voter "
               "may grant the vote, returning grant with the candidate term, or a refusal with "
               "the stale voted-for term or the voter's exposed log index as the witness.")
    config_cls = RaftVoteEligibilityConfig

    def generate_entry(self):
        while True:
            cfg = self.config
            max_term = cfg.max_term
            max_index = cfg.max_index
            n_c = cfg.n_candidates

            mode = random.choice(["grant", "stale", "log"])

            voter_id = random.randint(10, 99)
            cand_ids = [random.randint(10, 99) for _ in range(n_c)]
            while voter_id in cand_ids:
                voter_id = random.randint(10, 99)

            Tc = random.randint(1, max_term)
            Lc = random.randint(1, max_term)
            Ic = random.randint(0, max_index)

            if mode == "grant":
                Vv = random.randint(0, Tc - 1)
                if random.random() < 0.5 and Lc > 1:
                    Lv = random.randint(1, Lc - 1)
                    Iv = random.randint(0, max_index)
                else:
                    Lv = Lc
                    Iv = random.randint(0, Ic)
            elif mode == "stale":
                Vv = random.randint(Tc, max_term)
                Lv = random.randint(1, max_term)
                Iv = random.randint(0, max_index)
            else:  # log
                Vv = random.randint(0, Tc - 1)
                if random.random() < 0.5 and Lc < max_term:
                    Lv = random.randint(Lc + 1, max_term)
                    Iv = random.randint(0, max_index)
                elif random.random() < 0.5 and Ic < max_index:
                    Lv = Lc
                    Iv = random.randint(Ic + 1, max_index)
                else:
                    Lv = Lc
                    Iv = Ic + 1
                    if Iv > max_index:
                        Lv = max_term
                        Iv = random.randint(0, max_index)

            dmode, dwit = _decide(Tc, Lc, Ic, Vv, Lv, Iv)
            if dmode != mode:
                continue
            if dwit < 0:
                continue

            payload = {
                "candidates": sorted(cand_ids),
                "terms": {str(c): random.randint(1, max_term) for c in cand_ids},
                "current_leader": cand_ids[0],
                "candidate_term": Tc,
                "leader_log": (Lc, Ic),
                "voter": voter_id,
                "voter_last_voted": Vv,
                "voter_log": (Lv, Iv),
            }
            fmt = {"grant": "grant", "stale": "stale", "log": "log"}
            answer = f"{fmt[dmode]}={dwit}"
            metadata = edict({
                "mode": mode,
                "candidate_term": Tc,
                "leader_log_term": Lc,
                "leader_log_index": Ic,
                "voter_last_voted": Vv,
                "voter_log_term": Lv,
                "voter_log_index": Iv,
                "witness": dwit,
            })
            metadata.payload = payload
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        terms = " ; ".join(
            f"candidate {c} runs at term {p['terms'][str(c)]}" for c in p["candidates"]
        )
        lc, lic = p["leader_log"]
        lv, liv = p["voter_log"]
        return (
            f"In a Raft cluster, the candidates are {p['candidates']}; {terms}. "
            f"The current leader is {p['current_leader']} with candidate term {p['candidate_term']} "
            f"and last log entry (term {lc}, index {lic}). "
            f"Voter {p['voter']} was last recorded voting for term {p['voter_last_voted']} and has "
            f"last log entry (term {lv}, index {liv}). "
            f"Per the Raft log-matching safety rule, this voter may grant the vote to the leader "
            f"exactly when the leader's candidate term is strictly greater than the term the voter "
            f"last voted for (the request is not stale) and the leader's log is at least as "
            f"up-to-date as the voter's: leader log term greater than the voter's, or equal log "
            f"term with leader index at least the voter's index. Otherwise the voter refuses, "
            f"either because the vote is stale or because the leader's log is not up-to-date.\n\n"
            f"Decide whether voter {p['voter']} may grant the vote to {p['current_leader']}. "
            f"Answer exactly one of three forms: \"grant=<candidate term>\", "
            f"\"stale=<term the voter last voted for>\", or \"log=<voter's last log index>\", "
            f"quoting the witness value after the '='."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        gt = entry.answer
        if a == gt:
            return 1.0
        al = a.lower()
        gt_mode = gt.split("=", 1)[0]
        try:
            num = int(a.split("=", 1)[1])
        except (ValueError, IndexError):
            return 0.0
        if al.startswith(gt_mode + "=") and num == int(gt.split("=", 1)[1]):
            return 1.0
        return 0.0
