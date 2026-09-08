import statistics
import sys
import time

from reasoning_core.tasks.regex import RegexInduction


def main(n=100):
    task = RegexInduction()
    synth_times = []
    answer_lens = []
    rejected = 0

    for _ in range(n):
        t0 = time.time()
        p = task.generate()
        dt = time.time() - t0
        if p is None:
            rejected += 1
            continue
        synth_times.append(dt)
        answer_lens.append(len(p.answer))

    print(f"attempts: {n}")
    print(f"success_rate: {len(synth_times) / n:.3f}")
    print(f"median_synthesis_time: {statistics.median(synth_times) if synth_times else 0:.4f}s")
    print(f"median_answer_length: {statistics.median(answer_lens) if answer_lens else 0}")
    print(f"rejected_examples: {rejected}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
