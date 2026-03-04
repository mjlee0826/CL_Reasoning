from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from Strategy.PromptAbstractFactory.PromptTranslateFactory import PromptTranslateFactory

from tqdm import tqdm

class Translate(Strategy):
    """
    Concrete strategy implementation for translation tasks.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log):
        super().__init__(config)
        
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        
        # Prevents IndexError if config.languages is not provided.
        if self.config.languages:
            self.config.displayName += f" ({self.config.languages[0]})"
        else:
            self.config.displayName += " (Auto)"

    def getPrompt(self, question: str) -> str:
        """
        Constructs the translation prompt using the Factory pattern.
        """
        # Safely get the target language or default to English
        target_lang = self.config.languages[0] if self.config.languages else "english"
        prompt = PromptTranslateFactory().getPrompt(target_lang, question)
        return prompt

    def getRes(self) -> list:
        """
        Executes the translation loop over the dataset and tracks progress.
        """
        self.log.logInfo(self, self.model, self.dataset)

        database = self.dataset.getData()
        result = [{
            "Model": self.model.config.to_dict(),
            "Dataset": self.dataset.config.to_dict(),
            "Strategy": self.config.to_dict()
        }]

        pbar = tqdm(total=self.dataset.config.dataNums)
        for data in database:
            translateQuestion = self.model.getRes(self.getPrompt(data["question"]))
            
            result.append({
                "id": data.get("id", "N/A"),
                "Question": data.get("question", ""),
                "Translated": translateQuestion
            })

            # Log the translated output
            self.log.logMessage(f'翻譯問題：\n{translateQuestion}')

            pbar.update()
        
        pbar.close()

        return result
    
    @staticmethod
    def getTokenLens(model: Model, data):
        """Calculate token usage for the translated text."""
        # Using .get() for safety in case 'Translated' key doesn't exist
        return model.getTokenLens(data.get("Translated", ""))