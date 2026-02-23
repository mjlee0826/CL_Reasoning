from Model.Model import Model
from Dataset.Dataset import Dataset
from Log.Log import Log

import re

class Strategy():
    def __init__(self):
        self.name: str = "Strategy"
    
    def printName(self):
        print(f'Strategy: {self.name}')

    def parseAnswer(self, answer: str) -> str:
        result: str = ""
        
        # Try to extract JSON format first: {"answer": "value"}
        try:
            json_match = re.findall(r'\{"answer":\s*"([^"]+)"\}', answer)
            if json_match:
                result = json_match[-1]
                return result
            
            # Fallback to original pattern matching
            match = re.search(r"[A-Za-z](?!.*[A-Za-z])", answer)
            if match:
                result = match.group(1).strip()
        except:
            result = ""
        return result
    
    def getName(self) -> str:
        return self.name

    def getRes(self, model: Model, dataset: Dataset, log: Log) -> list:
        return []
    
    @staticmethod
    def getTokenLens(model: Model, data):
        return 0