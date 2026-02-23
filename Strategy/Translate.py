from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyType import STRATEGY_TO_LANGUAGE
from Log.Log import Log
from Strategy.PromptAbstractFactory.PromptTranslateFactory import PromptTranslateFactory

from tqdm import tqdm

class Translate(Strategy):
    def __init__(self, model: Model, dataset: Dataset, log: Log, type):
        super().__init__()
        self.name: str = f'{STRATEGY_TO_LANGUAGE[type]} Tranlated'
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.type = type

    def getPrompt(self, question: str) -> str:
        prompt = PromptTranslateFactory().getPrompt(self.type, question)
        return prompt

    def getRes(self) -> list:
        self.log.logInfo(self, self.model, self.dataset)

        database = self.dataset.getData()
        result = [{
            "Model": self.model.getModelName(),
            "Dataset": self.dataset.getName(),
            "Strategy": self.name,
            "Data Nums": self.dataset.getNums(),
            "Data Samples": self.dataset.getSample()
        }]

        pbar = tqdm(total=self.dataset.getDataNums())
        for data in database:
            translateQuestion = self.model.getRes(self.getPrompt(data["question"]))
            result.append({
                "id": data["id"],
                "Question": data["question"],
                "Translated": translateQuestion
            })

            self.log.logMessage(f'翻譯問題：\n{translateQuestion}')

            pbar.update()
        
        pbar.close()

        return result
    
    @staticmethod
    def getTokenLens(model: Model, data):
        return model.getTokenLens(data["Translated"])