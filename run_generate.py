from argparse import ArgumentParser
import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from Strategy.RunContext import RunContext

from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Model.ModelFactory import ModelFactory
from Model.ModelType import MODEL_STR_LIST, ModelType

from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType

from Strategy.StrategyConfig import StrategyConfig
from Strategy.Generate import Generate
from Arm.ArmSpec import ArmSpec
from File.ResultStore import ResultStore

from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog

# Datasets that have the translation / rewrite pipeline wired (see Dataset/*.py)
ACTIVE_DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]


def parseArgs():
    parser = ArgumentParser(description="Generate arms (paper_status v7 §3.3): model × dataset × arm")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, nargs="+", help="Choose your model(s)")
    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, required=True, nargs="+", help="Choose your dataset(s)")
    parser.add_argument("--arms", required=True, nargs="+",
                        help="arm_ids, e.g. L:en L:ja S:T0.7:seed3 R:short_cot P:expert W:rewrite")
    parser.add_argument("--nums", default=-1, type=int,
                        help="Data Nums to evaluate (-1 for all). Use the legacy value (2000) so item_ids line up with the anchor")

    parser.add_argument("--outdir", default="result/arms", help="Root directory of the arm files")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max concurrent threads/workers")
    return parser.parse_args()


def armPath(outdir: str, model_name: str, dataset_name: str, arm: ArmSpec) -> str:
    return os.path.join(outdir, model_name, dataset_name, f"{arm.file_stem}.json")


def runArm(model_name: str, dataset_name: str, arm: ArmSpec, args):
    log = OneAgentLog() if args.log else NoLog()

    model: Model = ModelFactory().buildModel(
        ModelType(model_name),
        ModelConfig.from_dict({"modelType": model_name, "temperature": arm.temperature}),
    )
    dataset: Dataset = DatasetFactory().buildDataset(DatasetType(dataset_name), arm.to_dataset_config(dataset_name, args.nums))

    if not model or not dataset:
        print(f"Error: Failed to build {model_name} or {dataset_name}.")
        return

    path = armPath(args.outdir, model_name, dataset_name, arm)
    store = ResultStore(path)
    strategy_config = StrategyConfig.from_dict({
        "strategyType": "generate",
        "languages": [arm.language],
        "promptStyle": arm.promptStyle,
    })
    strategy = Generate(strategy_config, model, dataset, log, arm, store)

    context = RunContext()
    context.setStrategy(strategy)
    failed = context.runExperiment()

    status = "🎉 Complete" if not failed else f"⚠️ {len(failed)} API failures, rerun the same command to retry"
    print(f"{status}: {model_name} | {dataset_name} | {arm.arm_id} -> {path} ({len(store.records)} records)")


def main():
    args = parseArgs()

    # Parse every arm_id before launching threads so a typo fails immediately
    arms = [ArmSpec.from_arm_id(arm_id) for arm_id in dict.fromkeys(args.arms)]
    tasks = list(itertools.product(args.model, args.dataset, arms))

    print("🚀 Preparing arm generation...")
    print(f"Models: {args.model}")
    print(f"Datasets: {args.dataset}")
    print(f"Arms: {[arm.arm_id for arm in arms]}")
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {args.workers}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(runArm, m, d, arm, args) for m, d, arm in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {e}")

    print("\n✅ All generation tasks finished!")


if __name__ == "__main__":
    main()
