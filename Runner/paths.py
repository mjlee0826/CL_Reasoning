import os

from Arm.ArmSpec import ArmSpec

# Datasets with the arm / rewrite pipeline wired (see Dataset/*.py)
ACTIVE_DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]


def armPath(outdir: str, model_name: str, dataset_name: str, arm: ArmSpec) -> str:
    return os.path.join(outdir, model_name, dataset_name, f"{arm.file_stem}.json")


def aggregationPath(outdir: str, model_name: str, dataset_name: str, aggregator_id: str, arms: list[ArmSpec]) -> str:
    stems = "__".join(arm.file_stem for arm in arms)
    return os.path.join(outdir, model_name, dataset_name, f"{aggregator_id}__{stems}.json")
