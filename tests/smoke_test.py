import pytest
from reasoning_core import list_tasks, get_task
import time

def test_all_tasks_validate():
    failed = []
    tasks = list_tasks()
    print(f"{len(tasks)} tasks")
    for i, t in enumerate(tasks):
        t0 = time.time()
        print(f"{i + 1:>4} {t.ljust(30, '.')}", end=' ', flush=True)
        try:
            task = get_task(t)
            task.validate(n_samples=5)
            print(f"{time.time() - t0:.5f}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"EXCEPTION: {e}")
            failed.append(t)

    if failed:
        raise RuntimeError(f"Failed tasks: {failed}")


if __name__ == "__main__":
    test_all_tasks_validate()
