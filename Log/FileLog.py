from Log.Log import Log
from File.File import File

import json

class FileLog(Log):
    def __init__(self):
        super().__init__()
    
    def logInfo(self, file: File):
        print('=' * 30)
        print(f'Log Information')
        print('=' * 50)
        # 1. Log Strategy Configuration
        print("💡 Strategy Config:")
        # json.dumps with indent=4 formats the dictionary into a highly readable multiline string
        # ensure_ascii=False allows proper display of multi-byte characters (like Chinese/Japanese)
        print(json.dumps(file.getStrategyConfig().to_dict(), indent=4, ensure_ascii=False))
        print("-" * 50)
        
        # 2. Log Model Configuration
        print("🤖 Model Config:")
        print(json.dumps(file.getModelConfig().to_dict(), indent=4, ensure_ascii=False))
        print("-" * 50)
        
        # 3. Log Dataset Configuration
        print("📁 Dataset Config:")
        print(json.dumps(file.getDatasetConfig().to_dict(), indent=4, ensure_ascii=False))
        print('=' * 30)