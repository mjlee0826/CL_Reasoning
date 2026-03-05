from Log.Log import Log
from File.File import File
import json

class TwoAgentLog(Log):
    """
    A logger specifically designed for strategies that involve two distinct agents or files, 
    such as the Challenge (Debate) strategy.
    """
    def __init__(self):
        super().__init__()
    
    def logInfo(self, strategy, model, dataset, file1: File, file2: File):
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
        print("-" * 50)

        # 4. Log File 1 Information
        print("📄 Agent 1 (File 1) Info:")
        file1_info = {
            "path": file1.file_path,
            "languages": file1.getLanguage()
        }
        print(json.dumps(file1_info, indent=4, ensure_ascii=False))
        print("-" * 50)

        # 5. Log File 2 Information
        print("📄 Agent 2 (File 2) Info:")
        file2_info = {
            "path": file2.file_path,
            "languages": file2.getLanguage()
        }
        print(json.dumps(file2_info, indent=4, ensure_ascii=False))
        print('=' * 50)