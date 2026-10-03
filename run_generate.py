from argparse import ArgumentParser

from Model.ModelType import MODEL_STR_LIST

from Strategy.StrategyConfig import StrategyConfig
from Strategy.Generate import Generate
from Arm.ArmSpec import ArmSpec
from File.ResultStore import ResultStore

from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog

from Runner.paths import ACTIVE_DATASETS, armPath
from Runner.tasks import defaultWorkers, interleaveByModel, runTasks
from Runner.builders import buildModel, buildArmDataset, runStrategy


def parseArgs():
    parser = ArgumentParser(description="Generate arms (paper_status v7 §3.3): model × dataset × arm")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, nargs="+", help="Choose your model(s)")
    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, required=True, nargs="+", help="Choose your dataset(s)")
    parser.add_argument("--arms", required=True, nargs="+",
                        help="arm_ids, e.g. L:en L:ja S:T0.7:seed3 R:short_cot P:expert W:rewrite1 W:rewrite2 F:en "
                             "(an F arm needs its complete base arm L:{lang} in --outdir; run them in separate commands)")
    parser.add_argument("--nums", default=-1, type=int,
                        help="Data Nums to evaluate (-1 for all). Use the legacy value (2000) so item_ids line up with the anchor")

    parser.add_argument("--outdir", default="result/arms", help="Root directory of the arm files")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Max concurrent threads/workers (default: one per task)")
    return parser.parse_args()


def runArm(model_name: str, dataset_name: str, arm: ArmSpec, args):
    log = OneAgentLog() if args.log else NoLog()

    model = buildModel(model_name, arm.temperature)
    dataset = buildArmDataset(dataset_name, arm, args.nums)

    if not model or not dataset:
        print(f"Error: Failed to build {model_name} or {dataset_name}.")
        return

    path = armPath(args.outdir, model_name, dataset_name, arm)
    store = ResultStore(path)
    # A derived arm (F:{lang}) reads its base arm's outputs from the same outdir; generate the base arm first
    baseStore = ResultStore(armPath(args.outdir, model_name, dataset_name, ArmSpec.from_arm_id(arm.base_arm_id))) \
        if arm.is_derived else None
    strategy_config = StrategyConfig.from_dict({
        "strategyType": "generate",
        "languages": [arm.language],
        "promptStyle": arm.promptStyle,
    })
    strategy = Generate(strategy_config, model, dataset, log, arm, store, baseStore)

    status = runStrategy(strategy)
    print(f"{status}: {model_name} | {dataset_name} | {arm.arm_id} -> {path} ({len(store.records)} records)")


def main():
    args = parseArgs()

    # Parse every arm_id before launching threads so a typo fails immediately
    arms = [ArmSpec.from_arm_id(arm_id) for arm_id in dict.fromkeys(args.arms)]
    tasks = interleaveByModel(args.model, args.dataset, arms)
    workers = args.workers or defaultWorkers(len(tasks))

    print("🚀 Preparing arm generation...")
    print(f"Models: {args.model}")
    print(f"Datasets: {args.dataset}")
    print(f"Arms: {[arm.arm_id for arm in arms]}")
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {workers}\n")

    runTasks(runArm, tasks, workers, args)

    print("\n✅ All generation tasks finished!")


if __name__ == "__main__":
    main()
