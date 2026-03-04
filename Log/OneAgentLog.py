from Log.Log import Log
import json

class OneAgentLog(Log):
    """
    A concrete logging implementation for a single-agent evaluation setup.
    Outputs the configuration details of the Strategy, Model, and Dataset in a readable JSON format.
    """
    def __init__(self):
        super().__init__()
    
    def logInfo(self, strategy, model, dataset):
        """
        Logs the initialization information before starting the evaluation loop.
        Leverages the .to_dict() methods of the config dataclasses for dynamic and structured logging.
        """
        print('=' * 50)
        print(f'🚀 [Experiment Initialization Log]')
        print('=' * 50)
        
        # 1. Log Strategy Configuration
        print("💡 Strategy Config:")
        # json.dumps with indent=4 formats the dictionary into a highly readable multiline string
        # ensure_ascii=False allows proper display of multi-byte characters (like Chinese/Japanese)
        print(json.dumps(strategy.config.to_dict(), indent=4, ensure_ascii=False))
        print("-" * 50)
        
        # 2. Log Model Configuration
        print("🤖 Model Config:")
        print(json.dumps(model.config.to_dict(), indent=4, ensure_ascii=False))
        print("-" * 50)
        
        # 3. Log Dataset Configuration
        print("📁 Dataset Config:")
        print(json.dumps(dataset.config.to_dict(), indent=4, ensure_ascii=False))
        print('=' * 50)