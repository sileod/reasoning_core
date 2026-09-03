import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload


TASK_META = {'parent_source_id': None,
 'idea': 'Add market-clearing reasoning over an order book.',
 'hypothesis': 'S23',
 'changes': 'Ask for the clearing price or the traded quantity implied by '
            'stated bids and asks.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2836545685,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class MarketClearingConfig(Config):
    n_buyers: int = 4
    n_sellers: int = 4
    price_hi: int = 100
    tie: str = "lowest"

    def apply_difficulty(self, level):
        self.n_buyers = int(self.n_buyers * (1 + 0.5 * level))
        self.n_sellers = int(self.n_sellers * (1 + 0.5 * level))
        self.price_hi = int(self.price_hi * (1 + 0.3 * level))


def _qty_at(buyers, sellers, price):
    d = sum(b[1] for b in buyers if b[0] >= price)
    s = sum(sq[1] for sq in sellers if sq[0] <= price)
    return min(d, s)


def _clearing_prices(buyers, sellers, max_qty, hi):
    out = []
    for p in range(1, hi + 1):
        if _qty_at(buyers, sellers, p) >= max_qty:
            out.append(p)
    return out


def _alloc(buyer, buyers, price, qty):
    ranked = sorted(buyers, key=lambda b: -b[0])
    traded = 0
    for i, x in enumerate(ranked):
        if x is buyer:
            if buyer[0] < price or traded >= qty:
                return 0
            return min(buyer[1], qty - traded)
        if x[0] >= price:
            traded += min(x[1], qty - traded)


def _buyer_surplus(buyers, price, qty):
    total = 0
    for b in buyers:
        amt = _alloc(b, buyers, price, qty)
        if amt:
            total += (b[0] - price) * amt
    return total


def _seller_surplus(sellers, price, qty):
    ranked = sorted(sellers, key=lambda s: s[0])
    traded = 0
    total = 0
    for s in ranked:
        if traded >= qty:
            break
        if s[0] <= price:
            take = min(s[1], qty - traded)
            total += (price - s[0]) * take
            traded += take
    return total


class MarketClearing(Task):
    config_cls = MarketClearingConfig
    task_version = 2

    def generate_entry(self):
        tie = self.config.tie
        price_hi = self.config.price_hi
        for _ in range(400):
            nb = max(2, self.config.n_buyers)
            ns = max(2, self.config.n_sellers)
            buyers = [(random.randint(1, price_hi), random.randint(1, 5)) for _ in range(nb)]
            sellers = [(random.randint(1, price_hi), random.randint(1, 5)) for _ in range(ns)]

            best_p, best_q = None, -1
            for p in range(1, price_hi + 1):
                q = _qty_at(buyers, sellers, p)
                if q > best_q:
                    best_q = q
                    best_p = p
            if best_q <= 0:
                continue
            max_q = best_q
            prices = _clearing_prices(buyers, sellers, max_q, price_hi)
            if len(prices) < 1:
                continue
            chosen = min(prices) if tie == "lowest" else max(prices)
            if max_q <= 0:
                continue

            task = random.choice(["price", "qty", "surplus_buyer", "surplus_seller"])
            if task == "price":
                answer = chosen
                question = ("What is the uniform clearing price at which the market clears, "
                            "following the stated tie-breaking rule?")
            elif task == "qty":
                answer = max_q
                question = ("What is the total quantity traded at the clearing price, "
                            "following the stated tie-breaking rule?")
            elif task == "surplus_buyer":
                answer = _buyer_surplus(buyers, chosen, max_q)
                question = ("What is the total consumer surplus (sum over buyers of "
                            "(bid price minus clearing price) times units each actually "
                            "buys) at the clearing price?")
            else:
                answer = _seller_surplus(sellers, chosen, max_q)
                question = ("What is the total producer surplus (sum over sellers of "
                            "(clearing price minus ask price) times units each actually "
                            "sells) at the clearing price?")

            metadata = edict({
                "buyers": buyers,
                "sellers": sellers,
                "tie": tie,
                "task": task,
                "task_text": question,
                "_clearing_price": chosen,
                "_qty": max_q,
            })
            metadata.payload = {
                "buyers": _fmt_buyers(buyers),
                "sellers": _fmt_sellers(sellers),
            }
            return Entry(metadata=metadata, answer=str(answer))
        raise RuntimeError("market clearing: no valid draw after bounded attempts")

    def render_prompt(self, metadata):
        tie_txt = ("If more than one price clears the same maximum quantity, take the "
                   "LOWEST such price." if metadata.tie == "lowest"
                   else "If more than one price clears the same maximum quantity, take the "
                   "HIGHEST such price.")
        return (f"An order book for one good is as follows:\n"
                f"{render_payload(metadata.payload)}\n"
                f"Tie-breaking rule: {tie_txt}\n"
                f"{metadata.task_text}\n"
                f"The answer is a single integer.")

    def score_answer(self, answer, entry):
        try:
            val = float(str(answer).strip())
        except Exception:
            return 0.0
        return 1.0 if val == float(entry.answer) else 0.0


def _fmt_buyers(buyers):
    parts = []
    for i, (p, q) in enumerate(buyers, 1):
        parts.append(f"buyer {i} will buy up to {q} units at a price of at most {p} (per unit)")
    return "; ".join(parts)


def _fmt_sellers(sellers):
    parts = []
    for i, (p, q) in enumerate(sellers, 1):
        parts.append(f"seller {i} will sell up to {q} units at a price of at least {p} (per unit)")
    return "; ".join(parts)
