import json
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
    parser = ArgumentParser(description="Translate datasets to multiple languages for cross-lingual evaluation.")

    # Logging Configuration
    parser.add_argument("--log", action="store_true", 
                        help="Enable verbose logging output to the terminal.")

    # Model Configuration
    parser.add_argument("-m", "--model", choices=MODEL_LIST, required=True,
                        help="Specify the base model family to evaluate.")
    parser.add_argument("--modelName", type=str,
                        help="Specify a particular model version (e.g., gpt-4.1, qwen, deepseek, gemini).")
    parser.add_argument("--temperature", default=0.0, type=float, 
                        help="Set the sampling temperature for the LLM (default: 0.0 for deterministic/greedy decoding).")

    # Dataset Configuration
    parser.add_argument("-d", "--dataset", choices=DATASET_LIST, required=True,
                        help="Select the evaluation dataset to be translated.")
    parser.add_argument("--nums", default=-1, type=int, 
                        help="Limit the total number of data points to process (default: -1 to process the entire dataset).")
    parser.add_argument("--sample", default=1, type=int, 
                        help="Set the number of inference samples to generate per data point (default: 1).")

    # Strategy Configuration
    parser.add_argument("-s", "--strategy", choices=STRATEGY_LIST, required=True,
                        help="Choose the specific translation strategy or framework approach to apply.")

    # Output Configuration
    parser.add_argument("--dirpath", type=str, 
                        help="Specify the target directory path to save the translated experiment results.")

    args = parser.parse_args()
    return args


def main():
    # 1. Parse configuration from command line
    args = parseArgs()

    # 2. Initialize the Model using the Factory pattern
    if args.model:
        model: Model = ModelFactory().buildModel(
            args.model, 
            temperature=args.temperature, 
            modelName=args.modelName
        )
    
    # 3. Initialize the Dataset using the Factory pattern
    if args.dataset:
        dataset: Dataset = DatasetFactory().buildDataset(
            args.dataset, 
            nums=args.nums, 
            sample=args.sample
        )
    
    # 4. Setup Logging mechanism based on user flag
    log: Log = OneAgentLog() if args.log else NoLog()

    # 5. Configure the Execution Context with the selected Strategy
    context: RunContext = RunContext()
    context.setStrategy(Translate(model, dataset, log, args.strategy))

    # 6. Execute the translation experiment
    result = context.runExperiment()

    # 7. Handle output saving
    if not result:
        print("Experiment yielded no results. Exiting.")
        return
    
    # Construct the output file path based on model, dataset, and strategy names
    path = ""
    file_name = f'{args.model}_{args.dataset}_{args.strategy}_translated.json'
    
    if args.dirpath:
        path = f'{args.dirpath}/{file_name}'
    else:
        path = file_name
        
    # Export the results to a JSON file with UTF-8 encoding
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Results successfully saved to {path}")

if __name__ == '__main__':
    main()