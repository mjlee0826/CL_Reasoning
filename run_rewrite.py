from argparse import ArgumentParser
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from Strategy.RunContext import RunContext

from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Model.ModelFactory import ModelFactory
from Model.ModelType import MODEL_STR_LIST, ModelType

from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType
from Dataset.path import rewriteFileName

from Strategy.StrategyConfig import StrategyConfig
from Strategy.Rewrite import Rewrite
from File.ResultStore import ResultStore

from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog

# Datasets that have the rewrite / prompt-variant pipeline wired (see Dataset/*.py).
ACTIVE_DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]


def parseArgs():
    parser = ArgumentParser(description="English question rewrite (paraphrase) generator")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    # A single fixed rewriter model produces one paraphrase file per dataset and version.
    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, help="The fixed rewriter model")
    parser.add_argument("--temperature", default=0.0, type=float, help="Model temperature setting")

    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, required=True, nargs="+", help="Dataset(s) to rewrite")
    parser.add_argument("--nums", help="Data Nums to rewrite (-1 for all; use the SAME value you will evaluate with)",
                        default=-1, type=int)
    parser.add_argument("--version", type=int, default=1,
                        help="Rewrite version. Version n >= 2 sees versions 1..n-1 and must word the question differently")

    parser.add_argument("--dirpath", help="Directory of the rewrite files", default="Data/rewritten")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Max concurrent threads/workers (default: one per task, i.e. full fan-out)")

    return parser.parse_args()


def loadPrevious(dirpath: str, dataset_name: str, version: int) -> dict:
    """{id: [rewrite version 1, ..., version - 1]} read from the earlier version files."""
    previous = {}
    for v in range(1, version):
        path = os.path.join(dirpath, rewriteFileName(dataset_name, v))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rewrite version {v} is needed before version {version}: {path}")
        with open(path, encoding="utf-8") as f:
            for record in json.load(f)[1:]:
                previous.setdefault(record["id"], []).append(record["Rewritten"])
    return previous


def runRewrite(model_name, dataset_name, args):
    log = OneAgentLog() if args.log else NoLog()

    model: Model = ModelFactory().buildModel(
        ModelType(model_name),
        ModelConfig.from_dict({"modelType": model_name, "temperature": args.temperature}),
    )

    dataset: Dataset = DatasetFactory().buildDataset(
        DatasetType(dataset_name),
        DatasetConfig.from_dict({
            "datasetType": dataset_name,
            "nums": args.nums,
            "sample": 1,
            "language": "english",   # rewrite operates on the original English text
        }),
    )

    if not model or not dataset:
        print(f"Error: Failed to build {model_name} or {dataset_name}.")
        return

    # Version n needs every earlier version for every item, otherwise "different from version 1" is undefined
    previous = loadPrevious(args.dirpath, dataset_name, args.version)
    if args.version >= 2:
        incomplete = [d["id"] for d in dataset.getData() if len(previous.get(d["id"], [])) != args.version - 1]
        if incomplete:
            raise ValueError(f"{dataset_name}: earlier rewrite versions miss {len(incomplete)} items; complete them first")

    path = os.path.join(args.dirpath, rewriteFileName(dataset_name, args.version))
    store = ResultStore(path, key="id")
    if store.metadata:
        found = (store.metadata.get("Model", {}).get("modelType"), store.metadata.get("Dataset", {}).get("nums"))
        expected = (model_name, dataset.config.nums)
        if found != expected:
            raise ValueError(f"{path} was written by another run: {found} != {expected}")

    strategy_config = StrategyConfig.from_dict({
        "strategyType": "rewrite",
        "languages": ["english"],
        "rewriteVersion": args.version,
    })
    strategy = Rewrite(strategy_config, model, dataset, log, store, previous)

    context = RunContext()
    context.setStrategy(strategy)
    failed = context.runExperiment()

    # A paraphrase identical to the question or to an earlier version adds no diversity
    records = store.records
    same_question = sum(r["Rewritten"].strip() == r["Question"].strip() for r in records.values())
    same_previous = sum(r["Rewritten"].strip() in [p.strip() for p in previous.get(i, [])] for i, r in records.items())
    status = "🎉 Complete" if not failed else f"⚠️ {len(failed)} API failures, rerun the same command to retry"
    print(f"{status}: {dataset_name} v{args.version} -> {path} ({len(records)} records, "
          f"identical to the question: {same_question}, identical to an earlier version: {same_previous})")


def main():
    args = parseArgs()

    tasks = list(itertools.product([args.model], args.dataset))
    workers = args.workers if args.workers is not None else len(tasks)
    workers = max(1, workers)

    print("🚀 Preparing English rewrite...")
    print(f"Rewriter model: {args.model} | Version: {args.version}")
    print(f"Datasets: {args.dataset}")
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {workers}\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(runRewrite, m, d, args) for m, d in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {e}")

    print("\n✅ All rewrite jobs finished!")


if __name__ == "__main__":
    main()
