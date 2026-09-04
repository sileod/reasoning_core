import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'two_phase_commit_trace (draw 1 of 1)',
 'hypothesis': 'HV-078',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/two_phase_commit_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2867049114,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class TwoPhaseCommitConfig(Config):
    n_participants: int = 4
    max_failures: int = 2

    def apply_difficulty(self, level):
        self.n_participants = int(self.n_participants) + level
        self.max_failures = min(self.n_participants, 2 + level)


def _protocol(votes, coord_commits, fail_set):
    """Final ('commit'/'abort') state of every participant under two-phase commit.

    Coordinator 'commit' vs 'abort': if decided commit it sends commit to every
    participant that voted yes (and abort to every participant that voted no or
    whose reply was missing). If decided abort it sends abort to everyone. A
    participant that replies nothing (it failed before replying) can never
    receive commit, so it aborts. Nothing depends on the coordinator failing
    after already logging/sending its decision.
    """
    outcomes = {}
    for p, v in votes.items():
        if v != 'yes' or not coord_commits:
            outcomes[p] = 'abort'
        else:
            outcomes[p] = 'commit'
    return outcomes


def _verify(participants, final, votes, coord_commits, fail_set):
    expected = _protocol(votes, coord_commits, fail_set)
    return all(final[p] == expected[p] for p in participants)


class TwoPhaseCommitTrace(Task):
    summary = ("Simulate two-phase commit over prepare/vote/commit/abort rounds with stated "
               "coordinator or participant failures and report the final commit/abort state of "
               "every participant.")
    config_cls = TwoPhaseCommitConfig

    def generate_entry(self):
        cfg = self.config
        participants = list(range(int(cfg.n_participants)))
        max_fail = int(cfg.max_failures)

        for _ in range(300):
            n_fail = random.randint(1, min(max_fail, len(participants)))
            fail_set = set(random.sample(participants, n_fail))
            coord_commits = random.random() < 0.5
            coord_dies_after = random.random() < 0.4
            votes = {}
            for p in participants:
                if p in fail_set:
                    votes[p] = None
                else:
                    votes[p] = random.choice(['yes', 'no'])

            final = _protocol(votes, coord_commits, fail_set)
            if not _verify(participants, final, votes, coord_commits, fail_set):
                continue
            if any(final[p] == 'commit' for p in participants):
                break

        answer = "".join("C" if final[p] == "commit" else "A" for p in participants)
        votes_str = {str(p): (str(v) if v is not None else "none") for p, v in votes.items()}
        final_str = {str(p): str(v) for p, v in final.items()}
        metadata = edict({
            "n_participants": len(participants),
            "participants": participants,
            "coord_commits": bool(coord_commits),
            "coord_dies_after": bool(coord_dies_after),
            "votes": votes_str,
            "failures": sorted(int(x) for x in fail_set),
            "final": final_str,
        })
        metadata.payload = {
            "participants": participants,
            "coord_commits": bool(coord_commits),
            "coord_dies_after": bool(coord_dies_after),
            "votes": votes_str,
            "failures": metadata.failures,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        dl = metadata.payload["participants"][0]
        fails = ", ".join(
            "participant %d" % x for x in metadata.payload["failures"]
        ) if metadata.payload["failures"] else "no participant fails"
        cstate = ("The coordinator stays alive throughout." if not metadata.payload["coord_dies_after"]
                  else "The coordinator crashes right after logging and dispatching every "
                       "decision, so each participant still receives the message the "
                       "coordinator had already sent to it.")
        vlines = "; ".join(
            "participant %d %s" % (p, metadata.payload["votes"][str(p)])
            for p in metadata.payload["participants"]
        )
        return (
            "Two-phase commit runs with one coordinator and participants %s. "
            "The coordinator sends prepare to every participant. Each participant replies "
            "%s (a participant that fails replies nothing and the coordinator treats a "
            "missing reply as no). After gathering the replies the coordinator sends commit "
            "to a participant exactly when that participant's reply was yes, and sends "
            "abort exactly when its reply was no or missing. %s %s. "
            "A participant ends committed only if it receives a commit message; in every "
            "other case it ends aborted. Report the final state of all participants as a "
            "string of C (commit) and A (abort) in participant order %s (example: AAC)."
            % (
                metadata.payload["participants"],
                vlines,
                cstate,
                fails,
                metadata.payload["participants"],
            )
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        answer = answer.strip().strip("'\"")
        if answer == entry.answer:
            return 1.0
        return 0.0

    def distractor_candidates(self, entry):
        n = entry.metadata.n_participants
        out = set()
        for i in range(n):
            cand = list(entry.answer)
            cand[i] = 'A' if entry.answer[i] == 'C' else 'C'
            out.add("".join(cand))
        return out
