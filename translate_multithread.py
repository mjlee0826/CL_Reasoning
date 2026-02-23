import os
import json
import itertools
import concurrent.futures
from argparse import ArgumentParser

# Import context and strategy execution framework
from Strategy.RunContext import RunContext
from Strategy.StrategyType import STRATEGY_LIST, StrategyType
from Strategy.Translate import Translate

# Import Model components
from Model.Model import Model
from Model.ModelFactory import ModelFactory
from Model.ModelType import MODEL_LIST

# Import Dataset components
from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DATASET_LIST

# Import Logging components
from Log.Log import Log
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog


def parseArgs():
    """
    Parses command-line arguments for the multilingual translation pipeline.
    """
    parser = ArgumentParser(description="Translate datasets to multiple languages concurrently.")

    # Logging Configuration
    parser.add_argument("--log", action="store_true", 
                        help="Enable verbose logging output to the terminal.")

    # Threading Configuration
    parser.add_argument("--workers", default=4, type=int,
                        help="Maximum number of concurrent threads to run experiments (default: 4).")

    # Model Configuration
    parser.add_argument("-m", "--model", choices=MODEL_LIST, required=True,
                        help="Specify the base model family to evaluate.")
    parser.add_argument("--modelName", type=str,
                        help="Specify a particular model version (e.g., gpt-4.1, qwen, deepseek, gemini).")
    parser.add_argument("--temperature", default=0.0, type=float, 
                        help="Set the sampling temperature for the LLM (default: 0.0).")

    # Dataset Configuration (Modified: nargs='+' to accept multiple datasets)
    parser.add_argument("-d", "--dataset", choices=DATASET_LIST, required=True, nargs='+',
                        help="Select one or more evaluation datasets to be translated (space-separated).")
    parser.add_argument("--nums", default=-1, type=int, 
                        help="Limit the total number of data points to process (default: -1).")
    parser.add_argument("--sample", default=1, type=int, 
                        help="Set the number of inference samples to generate per data point (default: 1).")

    # Strategy Configuration (Modified: nargs='+' to accept multiple strategies)
    parser.add_argument("-s", "--strategy", choices=STRATEGY_LIST, required=True, nargs='+',
                        help="Choose one or more translation strategies to apply (space-separated).")

    # Output Configuration
    parser.add_argument("--dirpath", type=str, default=".",
                        help="Specify the target directory path to save the translated experiment results.")

    args = parser.parse_args()
    return args


def run_single_experiment(model_base, model_name, temperature, dataset_name, nums, sample, strategy_name, log_enabled, dirpath):
    """
    Worker function to execute a single combination of Model + Dataset + Strategy.
    Instances are created locally to ensure thread safety.
    """
    try:
        # 1. Initialize Thread-Local Instances
        model: Model = ModelFactory().buildModel(
            model_base, 
            temperature=temperature, 
            modelName=model_name
        )
        
        dataset: Dataset = DatasetFactory().buildDataset(
            dataset_name, 
            nums=nums, 
            sample=sample
        )
        
        log: Log = OneAgentLog() if log_enabled else NoLog()

        # 2. Setup Context
        context: RunContext = RunContext()
        context.setStrategy(Translate(model, dataset, log, strategy_name))

        # 3. Execute
        result = context.runExperiment()

        if not result:
            return f"[Warning] Experiment yielded no results for: {dataset_name} with {strategy_name}"

        # 4. Save Output
        file_name = f'{model_base}_{dataset_name}_{strategy_name}_translated.json'
        path = os.path.join(dirpath, file_name)
        
        # Ensure output directory exists
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        return f"[Success] Results saved to {path}"
        
    except Exception as e:
        return f"[Error] Failed on {dataset_name} with {strategy_name}: {str(e)}"


def main():
    args = parseArgs()

    # Generate all combinations of datasets and strategies
    # For example: 2 datasets * 3 strategies = 6 tasks
    tasks = list(itertools.product(args.dataset, args.strategy))
    total_tasks = len(tasks)
    
    print(f"Starting experiment pipeline...")
    print(f"Target Model: {args.model} ({args.modelName})")
    print(f"Total Combinations to run: {total_tasks}")
    print(f"Concurrent Workers: {args.workers}")
    print("-" * 50)

    # Execute tasks using a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks to the thread pool
        future_to_task = {
            executor.submit(
                run_single_experiment,
                args.model,
                args.modelName,
                args.temperature,
                ds_name,
                args.nums,
                args.sample,
                strat_name,
                args.log,
                args.dirpath
            ): (ds_name, strat_name) 
            for ds_name, strat_name in tasks
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            ds_name, strat_name = future_to_task[future]
            try:
                # Get the return string from the worker function
                status_message = future.result()
                print(status_message)
            except Exception as exc:
                print(f"[Fatal Error] Task {ds_name}_{strat_name} generated an exception: {exc}")

if __name__ == '__main__':
    main()