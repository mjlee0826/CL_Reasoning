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
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

def parseArgs():
    parser = ArgumentParser(description="IMSR Evaluation Framework")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    # 將 nargs 設定為 '+', 讓這些參數可以接收多個值，變成一個 List
    parser.add_argument("-m", "--model", choices=MODEL_STR_LIST, required=True, nargs='+', help="Choose your model(s)")
    parser.add_argument("--temperature", default=0.0, type=float, help="Model temperature setting")

    parser.add_argument("-d", "--dataset", choices=DATASET_STR_LIST, required=True, nargs='+', help="Choose your dataset(s)")
    parser.add_argument("--nums", help="Data Nums to evaluate (-1 for all)", default=-1, type=int)
    parser.add_argument("--sample", help="Data Sample multiplier", default=1, type=int)

    parser.add_argument("-l", "--language", choices=LANGUAGE_STR_LIST, required=True, nargs='+', help="Choose the target language(s)")
    parser.add_argument("--dirpath", help="Directory path to save results", default=".")
    
    # 新增 workers 參數，讓你可以控制最大執行緒數量，避免踩到 API Rate Limit
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max concurrent threads/workers")

    args = parser.parse_args()
    return args

def runExperiment(model_name, dataset_name, language, args):
    """
    單一實驗的執行函式。每個執行緒都會跑一次這個函式。
    """
    # 建立 Log
    log = OneAgentLog() if args.log else NoLog()

    # Build Model
    modelFactory = ModelFactory()
    model_config = {
        'modelType': model_name,
        'temperature': args.temperature
    }
    model: Model = modelFactory.buildModel(ModelType(model_name), ModelConfig.from_dict(model_config))

    # Build Dataset 
    datasetFactory = DatasetFactory()
    dataset_config = {
        'datasetType': dataset_name,
        'nums': args.nums,
        'sample': args.sample,
        'language': language
    }
    dataset: Dataset = datasetFactory.buildDataset(DatasetType(dataset_name), DatasetConfig.from_dict(dataset_config))

    if not model or not dataset:
        print(f"Error: Failed to build {model_name} or {dataset_name}.")
        return

    # Build Strategy
    strategy_config_dict = {
        'strategyType': 'onelanguage',
        'languages': [language]
    }
    strategy_config = StrategyConfig.from_dict(strategy_config_dict)
    strategy = OnlyOneLanguage(strategy_config, model, dataset, log)

    # Run Execution Context
    context = RunContext()
    context.setStrategy(strategy)
    result = context.runExperiment()

    if not result:
        print(f"Experiment {model_name} - {dataset_name} - {language} yielded no results.")
        return

    # Save Results
    if args.dirpath and not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath, exist_ok=True) # exist_ok=True 避免多執行緒同時建立資料夾報錯

    file_name = f'{model_name}_{dataset_name}_onelanguage_{language}.json'
    path = os.path.join(args.dirpath, file_name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"🎉 Success! Results saved to: {path}")

def main():
    args = parseArgs()
    
    # 產生所有參數的排列組合 (Cartesian Product)
    tasks = list(itertools.product(args.model, args.dataset, args.language))
    
    print("🚀 Preparing Multithreaded Experiments...")
    print(f"Total combinations to run: {len(tasks)}")
    print(f"Concurrent workers: {args.workers}\n")

    # 啟動 ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for model_name, dataset_name, language in tasks:
            # 將任務提交給執行緒池
            futures.append(
                executor.submit(runExperiment, model_name, dataset_name, language, args)
            )

        # 捕捉並回報任何在執行緒中發生的 Exception
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {e}")

    print("\n✅ All experiments finished!")

if __name__ == '__main__':
    main()