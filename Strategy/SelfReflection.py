from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from File.File import File

from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Strategy.PromptAbstractFactory.PromptSelfReflectionCOTFactory import PromptSelfReflectionCOTFactory

from tqdm import tqdm

class SelfReflection(Strategy):
    """
    Implements a 'Self-Reflection' strategy.
    The model is provided with its previous answer and asked to reflect, critique, 
    and potentially correct its own logic to generate a better final answer.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, file: File):
        super().__init__(config)
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.file: File = file

        # Safely extract the language used in the previous run from the File
        langs = self.file.getLanguage()
        self.target_lang = langs[0] if isinstance(langs, list) and langs else "english"

        # Dynamically update the display name to reflect the language
        self.config.displayName += f"{self.target_lang}"

    def getPrompt(self) -> str:
        """
        Constructs the self-reflection prompt and the expected output format.
        """
        prompt = PromptSelfReflectionCOTFactory().getPrompt(self.target_lang) \
                + PromptFormatFactory().getPrompt(self.target_lang)
        return prompt

    def getRes(self) -> list:
        """
        Executes the self-reflection evaluation loop.
        Aligns the previous model responses with the dataset using O(1) ID lookup,
        and constructs a conversational context for the model to reflect upon.
        """
        self.log.logInfo(self, self.model, self.dataset)

        # 1. Prepare Metadata Header
        result = [{
            "Model": self.model.config.to_dict(),
            "Dataset": self.dataset.config.to_dict(),
            "Strategy": self.config.to_dict()
        }]

        database = self.dataset.getData()
        reflection_ids = []

        # 2. Pre-calculate valid records for an accurate progress bar
        for data in database:
            q_id = data.get("id")
            prev_record = self.file.getRecordById(q_id)
            if prev_record and prev_record.get("Result"):
                reflection_ids.append(q_id)

        self.log.logMessage(f'Total Valid Records for Reflection: {len(reflection_ids)} / {len(database)}')
        pbar = tqdm(total=len(reflection_ids), desc="Reflecting")

        # 3. Main Evaluation Loop
        for data in database:
            q_id = data.get("id")
            prev_record = self.file.getRecordById(q_id)

            # Skip if there is no previous record to reflect upon
            if not prev_record or not prev_record.get("Result"):
                continue

            # The current question is ALREADY translated thanks to Dataset Double Loading
            current_question = data.get("question", "")
            
            # The full chain-of-thought output from the previous run
            prev_output = prev_record.get("Result", "")

            # Construct the conversational history to mimic a back-and-forth self-critique
            chat_record = [
                {"role": "user", "content": current_question},
                {"role": "assistant", "content": prev_output},
                {"role": "user", "content": self.getPrompt()}
            ]

            # Ask the model to generate a new result based on the chat history
            resultAnswer = self.model.getListRes(chat_record)
            my_answer = self.parseAnswer(resultAnswer)

            result.append({
                "id": q_id,
                "Question": current_question,
                "Response": prev_output,  # The original flawed/initial output
                "Result": resultAnswer,   # The new reflected output
                "Answer": data.get("answer", ""),
                "MyAnswer": my_answer
            })

            # Log interactions for debugging
            self.log.logMessage(f'問題 (Question)：\n{current_question}')
            self.log.logMessage(f'前次輸出 (Previous Response)：\n{prev_output}')
            self.log.logMessage(f'反思結果 (Reflected Result)：\n{resultAnswer}')
            self.log.logMessage(f'My Answer: {my_answer} | Correct Answer: {data.get("answer", "")}\n')

            pbar.update()
        
        pbar.close()

        return result
    
    @staticmethod
    def getTokenLens(model: Model, data):
        """
        Calculates the total token usage for the reflection interaction.
        Accounts for the original question, the previous response fed back to the model,
        and the newly generated reflection result.
        """
        # Summing the lengths of the components involved in the conversational prompt
        # Multipliers can be applied if you are simulating the full chat history cost
        tokens = model.getTokenLens(data.get("Question", ""))
        tokens += model.getTokenLens(data.get("Response", "")) * 2 # Represents reading it back
        tokens += model.getTokenLens(data.get("Result", ""))
        return tokens