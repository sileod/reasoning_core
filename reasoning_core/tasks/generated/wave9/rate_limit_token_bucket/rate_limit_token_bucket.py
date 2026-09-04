import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class RateLimitTokenBucketConfig(Config):
    max_requests: int = 4
    max_capacity: int = 8
    max_refill: int = 4
    max_rate: int = 3

    def apply_difficulty(self, level):
        self.max_requests = sround(self.max_requests + level * 2)
        self.max_capacity = sround(self.max_capacity + level * 3)
        self.max_refill = sround(self.max_refill + level * 2)
        self.max_rate = sround(self.max_rate + level)


def simulate(capacity, refill_rate, refill_every, initial_tokens, requests, rate):
    """Return (accepted_times, final_tokens). Requests: list of (time, need)."""
    tokens = float(initial_tokens)
    accepted = []
    for t, need in requests:
        tokens = min(float(capacity), tokens + (t // refill_every) * refill_rate)
        if tokens >= need:
            if rate is None:
                tokens -= need
                accepted.append(t)
            else:
                window_start = max(0, t - rate)
                window_need = sum(r[1] for r in requests if window_start <= r[0] <= t)
                if window_need <= rate:
                    tokens -= need
                    accepted.append(t)
    return accepted, tokens


def final_balance(capacity, refill_rate, refill_every, initial_tokens, requests):
    tokens = float(initial_tokens)
    for t, need in requests:
        tokens = min(float(capacity), tokens + (t // refill_every) * refill_rate)
        if tokens >= need:
            tokens -= need
    return int(tokens)


MODE_ACCEPT = "accept"
MODE_BALANCE = "balance"


class RateLimitTokenBucket(Task):
    summary = ("Process timestamped requests through exact token-bucket refill and consumption "
               "rules, returning accepted request times or the final token balance; modes vary "
               "the bucket refill semantics and per-request consumption.")
    config_cls = RateLimitTokenBucketConfig

    def generate_entry(self):
        cfg = self.config_cls
        mode = random.choice([MODE_ACCEPT, MODE_BALANCE])
        capacity = random.randint(1, cfg.max_capacity)
        refill_rate = random.randint(1, cfg.max_refill)
        refill_every = random.randint(1, 3)
        initial_tokens = random.randint(0, capacity)
        n = random.randint(2, cfg.max_requests)
        rate = random.randint(1, cfg.max_rate) if mode == MODE_ACCEPT else None

        while True:
            times = sorted(random.sample(range(refill_every, refill_every * (n + 2) + 2), n))
            needs = [random.randint(1, 3) for _ in range(n)]
            requests = list(zip(times, needs))
            if mode == MODE_ACCEPT:
                accepted, _ = simulate(capacity, refill_rate, refill_every, initial_tokens, requests, rate)
                valid = len(accepted) >= 1
            else:
                valid = True
            if valid:
                break

        if mode == MODE_ACCEPT:
            accepted, _ = simulate(capacity, refill_rate, refill_every, initial_tokens, requests, rate)
            answer = ",".join(str(t) for t in accepted)
        else:
            bal = final_balance(capacity, refill_rate, refill_every, initial_tokens, requests)
            answer = str(bal)

        metadata = edict({
            "mode": mode,
            "capacity": int(capacity),
            "refill_rate": int(refill_rate),
            "refill_every": int(refill_every),
            "initial_tokens": int(initial_tokens),
            "requests": [[int(t), int(nd)] for t, nd in requests],
            "rate": int(rate) if mode == MODE_ACCEPT else None,
        })
        metadata.payload = {
            "capacity": int(capacity),
            "refill_rate": int(refill_rate),
            "refill_every": int(refill_every),
            "initial_tokens": int(initial_tokens),
            "requests": [[int(t), int(nd)] for t, nd in requests],
            "rate": int(rate) if mode == MODE_ACCEPT else None,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = [
            "A token bucket rate limiter has a maximum fill capacity of "
            f"{metadata.capacity} tokens. Initially the bucket holds {metadata.initial_tokens} tokens. "
            f"Every {metadata.refill_every} time units the bucket is refilled by {metadata.refill_rate} tokens, "
            "stopping when it reaches capacity."
        ]
        req_text = []
        for t, nd in metadata.requests:
            req_text.append(f"at time {t} a request for {nd} tokens arrives")
        lines.append(", ".join(req_text) + ".")
        if metadata.mode == "accept":
            lines.append(
                f"Additionally, a sliding window of {metadata.rate} time units limits the total demand of "
                f"requests within any window of that length to at most {metadata.rate} tokens."
            )
        lines.append(
            "Process the requests in increasing time order. A request is accepted only if the bucket "
            "currently holds enough tokens for it"
        )
        second = ""
        if metadata.mode == "accept":
            lines[-1] += (
                f", AND the combined demand of all requests whose times lie within {metadata.rate} time units of "
                f"it (including itself) does not exceed {metadata.rate} tokens"
            )
        lines[-1] += ". Accepted requests consume their tokens from the bucket; rejected requests consume none."
        if metadata.mode == "accept":
            lines.append("List the times of the accepted requests, in increasing order, separated by commas.")
            lines.append("The answer is a comma-separated list of integers.")
        else:
            lines.append("What is the token balance of the bucket right after the last request is handled?")
            lines.append("The answer is an integer.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        meta = entry.metadata
        if meta.mode == "accept":
            try:
                parts = answer.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                toks = [p for p in parts.replace(",", " ").split() if p != ""]
                parsed_times = [int(p) for p in toks]
            except (ValueError, AttributeError):
                return 0.0
            accepted, _ = simulate(
                meta["capacity"], meta["refill_rate"], meta["refill_every"],
                meta["initial_tokens"], meta["requests"], meta["rate"],
            )
            return 1.0 if parsed_times == accepted else 0.0
        else:
            try:
                val = int(float(answer.strip()))
            except (ValueError, AttributeError):
                return 0.0
            bal = final_balance(
                meta["capacity"], meta["refill_rate"], meta["refill_every"],
                meta["initial_tokens"], meta["requests"],
            )
            return 1.0 if val == bal else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'rate_limit_token_bucket (draw 1 of 1)',
 'hypothesis': 'HV-045',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/rate_limit_token_bucket',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3326285053,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
