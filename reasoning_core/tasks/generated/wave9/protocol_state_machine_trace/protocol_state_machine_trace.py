"""Event-driven protocol state machine trace task.

Given a deterministic finite protocol machine with guarded transitions,
retries and timeouts, plus a supplied event trace, run the machine and return
the name of the state reached after processing the whole trace.

Machine semantics (fully documented in the prompt):
- States S, event types E. Each (state, event) transition lists a target, a
  retry limit `r`, and a timeout `t`. Events (and their transitions) are the
  machine's protocol actions.
- A transition into a target is guarded: it fires only if the set of
  "previous states" allowed by that target contains the state the machine was
  in when the event arrived.
- The timeout `t` is the maximum number of events that may be processed while
  the machine stays in the same state. Once the machine successfully changes
  state that counter resets to zero.
- Per arriving event the machine makes up to `r` attempts; an attempt fires the
  transition iff (a) the target's guard allows the previous state and
  (b) the events-since-last-change counter is below the transition's timeout.
  Because attempts are identical, the transition's fired/blocked outcome is
  unambiguous and the trace fully determines the final state.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


TASK_META = {'parent_source_id': None,
 'idea': 'protocol_state_machine_trace (draw 1 of 1)',
 'hypothesis': 'HV-076',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/protocol_state_machine_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4137643316,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ProtocolStateMachineConfig(Config):
    n_states: int = 3
    n_events: int = 3
    trace_len: int = 5
    max_retry: int = 2

    def apply_difficulty(self, level):
        self.n_states = 2 + (level % 2) + level // 2
        self.n_events = 2 + level // 2
        self.trace_len = 5 + level * 2
        self.max_retry = 2 + level // 2


_EVENT_KINDS = ("msg", "ack", "nack", "timeout", "req", "recv", "drop")


def _simulate(states, events, target_of, guard_of, timeout_of, start, trace):
    """Run the machine per the documented semantics. Returns final state."""
    state = start
    prev = start
    hops = 0
    for ev in trace:
        tgt = target_of[state][ev]
        t = timeout_of[state][ev]
        guard_ok = (not guard_of[tgt]) or (prev in guard_of[tgt])
        if guard_ok and hops < t:
            prev = state
            state = tgt
            hops = 0
        else:
            hops += 1
    return state


class ProtocolStateMachineTrace(Task):
    summary = ("Execute event-driven protocol transitions with guards, "
               "retries, and timeouts, returning the final state after a "
               "supplied event trace.")
    config_cls = ProtocolStateMachineConfig

    def __init__(self, config=None):
        super().__init__(config)

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_states
        n_e = cfg.n_events
        states = ["s%d" % i for i in range(n)]
        events = ["%s%d" % (random.choice(_EVENT_KINDS), i + 1)
                  for i in range(n_e)]

        target_of = {}
        retry_of = {}
        timeout_of = {}
        for s in states:
            target_of[s] = {}
            retry_of[s] = {}
            timeout_of[s] = {}
            for e in events:
                target_of[s][e] = random.choice(states)
                retry_of[s][e] = random.randint(1, cfg.max_retry)
                timeout_of[s][e] = random.randint(1, cfg.max_retry + 1)

        guard_of = {}
        for s in states:
            if random.random() < 0.6:
                guard_of[s] = set(random.sample(states,
                                                random.randint(1, len(states))))
            else:
                guard_of[s] = set()

        start = random.choice(states)
        trace = [random.choice(events) for _ in range(cfg.trace_len)]
        answer = _simulate(states, events, target_of, guard_of, timeout_of,
                           start, trace)
        assert answer in states

        body = self._build_body(states, events, target_of, retry_of,
                                timeout_of, guard_of, start)

        metadata = edict({
            "states": states,
            "events": events,
            "start": start,
            "trace": trace,
            "trace_len": cfg.trace_len,
            "target_of": {"%s|%s" % (s, e): target_of[s][e]
                          for s in states for e in events},
            "retry_of": {"%s|%s" % (s, e): retry_of[s][e]
                         for s in states for e in events},
            "timeout_of": {"%s|%s" % (s, e): timeout_of[s][e]
                           for s in states for e in events},
            "guard_of": {s: sorted(guard_of[s]) for s in states},
            "prompt_body": body,
            "machine": body,
            "trace_text": " ".join(trace),
            "final_state": answer,
        })
        metadata.payload = {
            "machine": body,
            "trace": " ".join(trace),
        }
        return Entry(metadata=metadata, answer=answer)

    def _build_body(self, states, events, target_of, retry_of, timeout_of,
                    guard_of, start):
        lines = []
        for s in states:
            parts = []
            for e in events:
                tgt = target_of[s][e]
                r = retry_of[s][e]
                t = timeout_of[s][e]
                parts.append("on %s -> %s (r=%d t=%d)"
                             % (e, tgt, r, t))
            gtxt = ""
            if guard_of[s]:
                gtxt = (" [requires prev in {%s}]"
                        % ", ".join(sorted(guard_of[s])))
            lines.append("state %s: %s%s." % (s, " ; ".join(parts), gtxt))
        lines.append("start %s" % start)
        return "\n".join(lines)

    def render_prompt(self, metadata):
        return (metadata.prompt_body +
                "\n\nevent trace: %s" % metadata.trace_text +
                "\n\nRun this protocol machine on the given event trace from "
                "the starting state. For each event, try the current state's "
                "transition for that event. The transition fires, changing "
                "state to its target, only if BOTH (a) the target's 'requires "
                "prev' set contains the previous state (an unguarded target "
                "always passes) AND (b) the number of events processed since "
                "the machine last changed state is less than that "
                "transition's timeout t. When a transition fires, reset that "
                "events-since-last-change counter to zero; otherwise "
                "increment it. The answer is the name of the state after the "
                "whole trace, e.g. 's0'.")

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        if isinstance(answer, str):
            return 1.0 if answer.strip() == entry.answer else 0.0
        return 0.0
