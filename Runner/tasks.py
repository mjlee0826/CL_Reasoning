import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed


def defaultWorkers(n_tasks: int) -> int:
    """Default thread count: one thread per task (at least 1)."""
    return max(1, n_tasks)


def interleaveByModel(models: list, *dims) -> list:
    """model × dims tasks ordered so that consecutive tasks rotate through the models, which spreads the
    threads over the providers from the start instead of giving them all to the first model."""
    return [(combo[-1], *combo[:-1]) for combo in itertools.product(*dims, models)]


def runTasks(fn, tasks: list[tuple], workers: int, *shared):
    """Runs fn(*task, *shared) for every task on `workers` threads; a task that raises is reported and the others go on."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, *task, *shared) for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {type(e).__name__}: {e}")
