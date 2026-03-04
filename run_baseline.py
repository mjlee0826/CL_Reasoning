from argparse import ArgumentParser
from Strategy.RunContext import RunContext

from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Model.ModelFactory import ModelFactory
from Model.ModelType import MODEL_STR_LIST, ModelType

from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DATASET_STR_LIST, DatasetType

from Strategy.StrategyConfig import StrategyConfig
from Strategy.StrategyType import STRATEGY_STR_LIST, LANGUAGE_STR_LIST
from Strategy.OnlyOneLanguage import OnlyOneLanguage
from Log.Log import Log
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog

import json
import os
import sys

def parseArgs():
    parser = ArgumentParser(description="IMSR Evaluation Framework")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    # Required arguments for Model, Dataset, and Language to prevent NoneType errors
    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, help="Choose your model")
    parser.add_argument("--temperature", default=0.0, type=float, help="Model temperature setting")

    parser.add_argument("-d", "--dataset", choices=DATASET_STR_LIST, required=True, help="Choose your dataset")
    parser.add_argument("--nums", help="Data Nums to evaluate (-1 for all)", default=-1, type=int)
    parser.add_argument("--sample", help="Data Sample multiplier", default=1, type=int)

    parser.add_argument("-l", "--language", choices=LANGUAGE_STR_LIST, required=True, help="Choose the target language")
    parser.add_argument("--dirpath", help="Directory path to save results", default=".")

    args = parser.parse_args()
    return args

def runExperiment(args):
    # 1. Initialize Logging
    log = OneAgentLog() if args.log else NoLog()

    # 2. Build Model
    modelFactory = ModelFactory()
    model_config = {
        'modelType': args.model,
        'temperature': args.temperature
    }
    print(ModelConfig.from_dict(model_config))
    model: Model = modelFactory.buildModel(ModelType(args.model), ModelConfig.from_dict(model_config))

    # 3. Build Dataset 
    datasetFactory = DatasetFactory()
    
    # 🚨 BUG FIX: Added 'language' to the config dictionary.
    # Without this, DatasetConfig would default to English and fail to load translated files!
    dataset_config = {
        'datasetType': args.dataset,
        'nums': args.nums,
        'sample': args.sample,
        'language': args.language  # <--- Added this crucial line!
    }
    dataset: Dataset = datasetFactory.buildDataset(DatasetType(args.dataset), DatasetConfig.from_dict(dataset_config))

    # Ensure instantiation was successful
    if not model or not dataset:
        print("Error: Failed to build Model or Dataset. Please check your arguments.")
        sys.exit(1)

    # 4. Build Strategy
    strategy_config_dict = {
        'strategyType': 'onelanguage',
        'languages': [args.language]
    }
    strategy_config = StrategyConfig.from_dict(strategy_config_dict)
    strategy = OnlyOneLanguage(strategy_config, model, dataset, log)

    # 5. Run Execution Context
    context = RunContext()
    context.setStrategy(strategy)
    result = context.runExperiment()

    if not result:
        print("Experiment yielded no results.")
        return

    # 6. Save Results
    # Ensure the target directory exists
    if args.dirpath and not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)

    # Use raw string types instead of display names to avoid spaces and brackets in filenames
    file_name = f'{args.model}_{args.dataset}_onelanguage_{args.language}.json'
    path = os.path.join(args.dirpath, file_name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Experiment finished successfully! Results saved to: {path}")

def main():
    args = parseArgs()
    print("🚀 Run Experiment Prepare...")
    runExperiment(args)

if __name__ == '__main__':
    main()