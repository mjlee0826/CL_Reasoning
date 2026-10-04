import os

from Arm.ArmSpec import ArmSpec

# Datasets with the arm / rewrite pipeline wired (see Dataset/*.py)
ACTIVE_DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]


def armPath(outdir: str, model_name: str, dataset_name: str, arm: ArmSpec) -> str:
    return os.path.join(outdir, model_name, dataset_name, f"{arm.file_stem}.json")


def aggregationPath(outdir: str, model_name: str, dataset_name: str, aggregator_id: str, arms: list[ArmSpec]) -> str:
    stems = "__".join(arm.file_stem for arm in arms)
    return os.path.join(outdir, model_name, dataset_name, f"{aggregator_id}__{stems}.json")


def crossJudgePath(outdir: str, judge_name: str, generator_name: str, dataset_name: str, arms: list[ArmSpec]) -> str:
    """RQ2 cross-judge file: {outdir}/{judge}/{generator}/{dataset}/judge__{arm}__{arm}.json"""
    return aggregationPath(os.path.join(outdir, judge_name), generator_name, dataset_name, "judge", arms)


def menuJudgePath(outdir: str, model_name: str, dataset_name: str, menu: str) -> str:
    """RQ1-KJ judge file of one menu: {outdir}/{model}/{dataset}/{menu}.json"""
    return os.path.join(outdir, model_name, dataset_name, f"{menu}.json")
