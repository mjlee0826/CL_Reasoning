import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# DeepSeek official API peak hours (UTC, Mon–Fri): double price
DEEPSEEK_PEAK_HOURS = set(range(1, 4)) | set(range(6, 10))


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


def warnDeepseekPeak(models: list[str]):
    """Prints a reminder when deepseek4.1flash is about to be called during its peak (double-price) hours."""
    now = datetime.now(timezone.utc)
    if "deepseek4.1flash" in models and now.weekday() < 5 and now.hour in DEEPSEEK_PEAK_HOURS:
        print(f"⚠️ {now:%H:%M} UTC is DeepSeek's peak period (Mon–Fri UTC 01–04, 06–10, double price); "
              f"consider running deepseek4.1flash off-peak\n")
