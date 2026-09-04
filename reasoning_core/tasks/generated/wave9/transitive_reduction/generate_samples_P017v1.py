import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.transitive_reduction.transitive_reduction import (
    TransitiveReduction,
    edges_to_answer,
    transitive_reduction,
    reachability_edges,
    _desc_payload,
)

SEED = 1958495724


def _list_example(task):
    n = task.config.n_nodes
    for _ in range(300):
        max_edges = n * (n - 1) // 2
        lo = max(n - 1, 2)
        hi = min(n * 3, max_edges)
        if hi < lo:
            hi = lo
        import reasoning_core.tasks.generated.wave9.transitive_reduction.transitive_reduction as M
        edges = M._random_dag(n, random.randint(lo, hi))
        red = transitive_reduction(edges, n)
        if reachability_edges(edges, n) == reachability_edges(list(red), n):
            payload = _desc_payload(edges, n)
            return payload, edges_to_answer(list(red))


def main():
    random.seed(SEED)
    task = TransitiveReduction()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"## Level {level}\n")
        for _ in range(2):
            payload, answer = _list_example(task)
            out.append("### Prompt")
            out.append("")
            out.append(task.render_prompt(payload))
            out.append("")
            out.append("### Answer")
            out.append("")
            out.append(answer)
            out.append("")
    path = Path(__file__).with_name("samples_P017v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
