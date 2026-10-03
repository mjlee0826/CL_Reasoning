from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Model.ModelFactory import ModelFactory
from Model.ModelType import ModelType

from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType

from Arm.ArmSpec import ArmSpec
from Strategy.Strategy import Strategy
from Strategy.RunContext import RunContext


def buildModel(model_name: str, temperature: float) -> Model:
    return ModelFactory().buildModel(
        ModelType(model_name), ModelConfig.from_dict({"modelType": model_name, "temperature": temperature})
    )


def buildArmDataset(dataset_name: str, arm: ArmSpec, nums: int) -> Dataset:
    """The questions an arm answers (its language and question source)."""
    return DatasetFactory().buildDataset(DatasetType(dataset_name), arm.to_dataset_config(dataset_name, nums))


def buildEnglishDataset(dataset_name: str, nums: int) -> Dataset:
    """The original English questions: what the judge sees in aggregation and what the rewriter rewrites."""
    return DatasetFactory().buildDataset(DatasetType(dataset_name), DatasetConfig.from_dict({
        "datasetType": dataset_name, "nums": nums, "sample": 1, "language": "english",
    }))


def runStrategy(strategy: Strategy) -> str:
    """Runs one strategy and returns the status that starts the task's summary line."""
    context = RunContext()
    context.setStrategy(strategy)
    failed = context.runExperiment()
    return "🎉 Complete" if not failed else f"⚠️ {len(failed)} API failures, rerun the same command to retry"
