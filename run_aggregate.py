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
from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType

from Strategy.StrategyConfig import StrategyConfig
from Strategy.Aggregate import Aggregate
from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorFactory import AggregatorFactory
from Aggregator.AggregatorType import AGGREGATOR_STR_LIST, AGGREGATOR_TO_K, AggregatorType
from Arm.ArmSpec import ArmSpec
from File.File import File
from File.ResultStore import ResultStore

from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog

from run_generate import ACTIVE_DATASETS, armPath


def parseArgs():
    parser = ArgumentParser(description="Aggregate arms (paper_status v7 §3.4): model × dataset × aggregator × candidate set")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, nargs="+", help="Choose your model(s)")
    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, required=True, nargs="+", help="Choose your dataset(s)")
    parser.add_argument("-a", "--aggregators", choices=AGGREGATOR_STR_LIST, required=True, nargs="+", help="aggregator_ids")
    parser.add_argument("--candidates", required=True, action="append",
                        help="Comma-separated arm_ids of one candidate set, e.g. L:en,L:ja (repeatable)")
    parser.add_argument("--nums", default=-1, type=int, help="Must equal the nums the arm files were generated with")

    parser.add_argument("--seed", type=int, default=0, help="Vote tie-breaks and judge presentation-order balancing")
    parser.add_argument("--threshold", type=int, default=3, help="Max debate rounds before the judge is called")

    parser.add_argument("--armdir", default="result/arms", help="Root directory of the arm files")
    parser.add_argument("--outdir", default="result/aggregations", help="Root directory of the aggregation files")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max concurrent threads/workers")
    return parser.parse_args()


def aggregationPath(outdir: str, model_name: str, dataset_name: str, aggregator_id: str, arms: list[ArmSpec]) -> str:
    stems = "__".join(arm.file_stem for arm in arms)
    return os.path.join(outdir, model_name, dataset_name, f"{aggregator_id}__{stems}.json")


def runAggregation(model_name: str, dataset_name: str, aggregator_id: str, arms: list[ArmSpec], args):
    log = OneAgentLog() if args.log else NoLog()
    arm_names = ",".join(arm.arm_id for arm in arms)

    paths = [armPath(args.armdir, model_name, dataset_name, arm) for arm in arms]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        print(f"⚠️ Skip [{model_name} | {dataset_name} | {aggregator_id} | {arm_names}]: missing arm files {missing}")
        return
    armFiles = [File(path) for path in paths]

    model: Model = ModelFactory().buildModel(
        ModelType(model_name), ModelConfig.from_dict({"modelType": model_name, "temperature": 0.0})
    )

    # Original English question (judge) + each arm's own question text; arms with the same source share a dataset
    datasetFactory = DatasetFactory()
    dataset: Dataset = datasetFactory.buildDataset(DatasetType(dataset_name), DatasetConfig.from_dict({
        "datasetType": dataset_name, "nums": args.nums, "sample": 1, "language": "english",
    }))
    sources = {("english", False): dataset}
    armDatasets = []
    for arm in arms:
        key = (arm.language, arm.questionSource == "rewrite")
        if key not in sources:
            sources[key] = datasetFactory.buildDataset(DatasetType(dataset_name), arm.to_dataset_config(dataset_name, args.nums))
        armDatasets.append(sources[key])

    aggregator = AggregatorFactory().buildAggregator(
        AggregatorType(aggregator_id),
        AggregatorConfig.from_dict({"seed": args.seed, "threshold": args.threshold}),
        model, dataset,
    )

    path = aggregationPath(args.outdir, model_name, dataset_name, aggregator_id, arms)
    store = ResultStore(path)
    strategy_config = StrategyConfig.from_dict({
        "strategyType": "aggregate",
        "languages": [arm.language for arm in arms],
    })
    strategy = Aggregate(strategy_config, model, dataset, log, aggregator, arms, armFiles, armDatasets, store)

    context = RunContext()
    context.setStrategy(strategy)
    failed = context.runExperiment()

    status = "🎉 Complete" if not failed else f"⚠️ {len(failed)} API failures, rerun the same command to retry"
    print(f"{status}: {model_name} | {dataset_name} | {aggregator_id} | {arm_names} -> {path} ({len(store.records)} records)")


def main():
    args = parseArgs()

    # Parse every candidate set before launching threads so a typo fails immediately
    candidate_sets = [[ArmSpec.from_arm_id(arm_id.strip()) for arm_id in group.split(",")] for group in args.candidates]

    tasks = []
    for m, d, aggregator_id, arms in itertools.product(args.model, args.dataset, args.aggregators, candidate_sets):
        k = AGGREGATOR_TO_K[AggregatorType(aggregator_id)]
        if len(arms) != k:
            print(f"⏭️ Skip {aggregator_id} for {[arm.arm_id for arm in arms]}: needs {k} candidates")
            continue
        tasks.append((m, d, aggregator_id, arms))

    print("🚀 Preparing aggregation...")
    print(f"Models: {args.model}")
    print(f"Datasets: {args.dataset}")
    print(f"Aggregators: {args.aggregators}")
    print(f"Candidate sets: {[[arm.arm_id for arm in arms] for arms in candidate_sets]}")
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {args.workers}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(runAggregation, m, d, a, arms, args) for m, d, a, arms in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {e}")

    print("\n✅ All aggregation tasks finished!")


if __name__ == "__main__":
    main()
