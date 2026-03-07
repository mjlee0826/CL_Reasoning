from argparse import ArgumentParser
import itertools
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from Strategy.Challenge import Challenge

from Log.Log import Log
from Log.NoLog import NoLog
from Log.TwoAgentLog import TwoAgentLog

from File.File import File  # 引入封裝好的 File 類別

def parseArgs():
    parser = ArgumentParser(description="IMSR Evaluation Framework - Challenge Mode")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")

    # 支援多個 Model 與 Dataset
    parser.add_argument("-m", "--modelType", choices=MODEL_STR_LIST, required=True, nargs='+', help="Choose your model(s)")
    parser.add_argument("--temperature", default=0.0, type=float, help="Model temperature setting")

    parser.add_argument("-d", "--datasetType", choices=DATASET_STR_LIST, required=True, nargs='+', help="Choose your dataset(s)")
    parser.add_argument("--nums", help="Data Nums to evaluate (-1 for all)", default=-1, type=int)
    parser.add_argument("--sample", help="Data Sample multiplier", default=1, type=int)

    # 語言參數設定為可複選，預設包含 5 種語言
    default_langs = ["english", "chinese", "japanese", "russian", "spanish"]
    parser.add_argument("-l", "--languages", choices=LANGUAGE_STR_LIST, nargs='+', default=default_langs, help="Languages to pair up for debate")
    
    # 辯論回合上限
    parser.add_argument("--threshold", type=int, default=3, help="Max debate turns before calling the judge")

    # 路徑與執行緒設定
    parser.add_argument("--input_dir", help="Directory containing the onelanguage JSON files", default="result/new_english_result")
    parser.add_argument("--output_dir", help="Directory to save challenge results", default="result/challenge_result")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max concurrent threads/workers")

    args = parser.parse_args()
    return args

def executeChallengeTask(model_name, dataset_name, lang1, lang2, args):
    """
    單一 Challenge 實驗的執行函式。負責讀取兩個對應的語言檔案，並讓它們進行辯論。
    """
    # 1. 組合輸入檔案路徑 (依照之前的命名慣例)
    file1_name = f'{model_name}_{dataset_name}_onelanguage_{lang1}.json'
    file2_name = f'{model_name}_{dataset_name}_onelanguage_{lang2}.json'
    path1 = os.path.join(args.input_dir, file1_name)
    path2 = os.path.join(args.input_dir, file2_name)

    # 檢查檔案是否存在，若缺少則略過這個組合
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"⚠️ Skip: Missing files for [{model_name} | {dataset_name}] ({lang1} vs {lang2})")
        return

    # 2. 載入檔案與 Log
    file1 = File(path1)
    file2 = File(path2)
    log = TwoAgentLog() if args.log else NoLog()

    # 3. 建立 Model 與 Dataset
    modelFactory = ModelFactory()
    model: Model = modelFactory.buildModel(ModelType(model_name), ModelConfig.from_dict({
        'modelType': model_name,
        'temperature': args.temperature
    }))

    datasetFactory = DatasetFactory()
    dataset: Dataset = datasetFactory.buildDataset(DatasetType(dataset_name), DatasetConfig.from_dict({
        'datasetType': dataset_name,
        'nums': args.nums,
        'sample': args.sample,
        'language': "english"  # 裁判 (Judge) 預設使用英文 Prompt
    }))

    if not model or not dataset:
        print(f"❌ Error: Failed to build {model_name} or {dataset_name}.")
        return

    # 4. 建立 Challenge Strategy
    strategy_config = StrategyConfig.from_dict({
        'strategyType': 'challenge',
        'languages': [lang1, lang2]
    })
    strategy = Challenge(strategy_config, model, dataset, log, file1, file2, args.threshold)

    # 5. 執行測試
    context = RunContext()
    context.setStrategy(strategy)
    result = context.runExperiment()

    if not result:
        print(f"⚠️ No results yielded for {model_name} - {dataset_name} ({lang1} vs {lang2})")
        return

    # 6. 儲存結果
    os.makedirs(args.output_dir, exist_ok=True)
    out_file_name = f'{model_name}_{dataset_name}_challenge_{lang1}_vs_{lang2}.json'
    out_path = os.path.join(args.output_dir, out_file_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"🎉 Success: {model_name} | {dataset_name} | {lang1} vs {lang2} -> {out_file_name}")

def main():
    args = parseArgs()
    
    # 利用 combinations 自動產生兩兩配對的組合 (例如 5 取 2 = 10 種)
    lang_pairs = list(itertools.combinations(args.languages, 2))
    
    # 產生所有的任務組合 (Model * Dataset * 10 種語言配對)
    tasks = []
    for m in args.modelType:
        for d in args.datasetType:
            for l1, l2 in lang_pairs:
                tasks.append((m, d, l1, l2))

    print("🚀 Preparing Challenge Experiments...")
    print(f"Models: {args.modelType}")
    print(f"Datasets: {args.datasetType}")
    print(f"Languages: {args.languages}")
    print(f"Total pairs per model/dataset: {len(lang_pairs)}")
    print(f"Total tasks to run: {len(tasks)}")
    print(f"Concurrent workers: {args.workers}\n")

    # 啟動 ThreadPoolExecutor 進行多執行緒辯論
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for model_name, dataset_name, lang1, lang2 in tasks:
            futures.append(
                executor.submit(executeChallengeTask, model_name, dataset_name, lang1, lang2, args)
            )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ A job generated an exception: {e}")

    print("\n✅ All Challenge experiments finished!")

if __name__ == '__main__':
    main()