from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from Strategy.PromptAbstractFactory.PromptCOTFactory import PromptCOTFactory
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory

from tqdm import tqdm

class OnlyOneLanguage(Strategy):
    """
    A strategy implementation that evaluates the model using a specific target language.
    
    Thanks to the Dataset Double Loading mechanism, the dataset questions are already
    translated. This strategy simply wraps the translated question with language-specific 
    Chain-of-Thought (COT) and Formatting prompts to generate and evaluate the answer.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log):
        super().__init__(config)
        
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        
        # Safely append the target language to the display name for clear reporting
        if self.config.languages:
            self.config.displayName += f" ({self.config.languages[0]})"
        else:
            self.config.displayName += " (Auto)"

    def getPrompt(self, question: str) -> str:
        """
        Constructs the reasoning prompt using the language-specific COT and Format factories.
        """
        target_lang = self.config.languages[0] if self.config.languages else "english"
        
        # Combine the Chain-of-Thought prompt and the Output Format prompt
        cot_prompt = PromptCOTFactory().getPrompt(target_lang, question)
        format_prompt = PromptFormatFactory().getPrompt(target_lang)
        
        return cot_prompt + format_prompt

    def getRes(self) -> list:
        """
        Executes the evaluation loop over the dataset and tracks progress.
        """
        self.log.logInfo(self, self.model, self.dataset)

        # The dataset will automatically return data in the target language
        # based on the DatasetConfig.language setting!
        database = self.dataset.getData()
        
        # Use the unified to_dict() method for modern configuration logging
        result = [{
            "Model": self.model.config.to_dict(),
            "Dataset": self.dataset.config.to_dict(),
            "Strategy": self.config.to_dict()
        }]

        pbar = tqdm(total=self.dataset.config.dataNums)
        for data in database:
            # The question is already translated by the Dataset, no need to call LLM for translation
            current_question = data["question"] 
            
            # Perform reasoning and get the result from the LLM
            resultAnswer = self.model.getRes(self.getPrompt(current_question))
            
            # Parse the actual answer letter/number from the LLM's raw output
            my_answer = self.parseAnswer(resultAnswer)
            
            result.append({
                "id": data.get("id", "N/A"),
                "Question": current_question,  # Logs the translated question
                "Result": resultAnswer,        # The full COT output from the model
                "Answer": data.get("answer", ""),  # The ground truth answer
                "MyAnswer": my_answer          # The extracted answer for evaluation
            })

            # Log the interactions for debugging and real-time monitoring
            self.log.logMessage(f'當前問題 (Question)：\n{current_question}')
            self.log.logMessage(f'模型輸出 (Result)：\n{resultAnswer}')
            self.log.logMessage(f'My Answer: {my_answer} | Correct Answer: {data.get("answer", "")}\n')

            pbar.update()
        
        pbar.close()

        return result
    
    @staticmethod
    def getTokenLens(model: Model, data):
        """
        Calculates token usage for the model's output.
        Note: We no longer calculate translation tokens here since the translation 
        is pre-computed and loaded directly by the Dataset.
        """
        return model.getTokenLens(data.get("Result", ""))