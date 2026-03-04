from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.OnlyOneLanguage import OnlyOneLanguage
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from File.File import File

from tqdm import tqdm

class Repair(OnlyOneLanguage):
    """
    A strategy designed to 'repair' incomplete results from a previous run.
    It reads data from a File and selectively re-processes items where 'MyAnswer' is empty or missing.
    Inherits from OnlyOneLanguage to elegantly reuse the prompt generation logic.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, file: File):
        # Initialize the parent class (OnlyOneLanguage) to inherit prompt and logic settings
        super().__init__(config, model, dataset, log)
        
        self.file: File = file
        
        # Override the display name to explicitly indicate it's running in Repair mode
        target_lang = self.config.languages[0] if self.config.languages else "Auto"
        self.config.displayName = f"Repair ({target_lang})"

    def getRes(self) -> list:
        """
        Executes the repair loop. It iterates through the target dataset, checks if a valid answer
        already exists in the loaded file, and only triggers the LLM for missing/empty answers.
        """
        self.log.logInfo(self, self.model, self.dataset)
        self.log.logMessage("Repair Mode: Only processing data with missing or empty 'MyAnswer'.")

        # 1. Prepare Metadata Header using the new Config to_dict() mechanism
        result = [{
            "Model": self.model.config.to_dict(),
            "Dataset": self.dataset.config.to_dict(),
            "Strategy": self.config.to_dict()
        }]

        database = self.dataset.getData()
        repair_ids = []
        
        # 2. Pre-calculate how many items actually need repairing using O(1) File lookup
        for data in database:
            q_id = data.get("id")
            record = self.file.getRecordById(q_id)
            
            # Identify records that are missing entirely or have an empty "MyAnswer"
            if not record or not record.get("MyAnswer"):
                repair_ids.append(q_id)

        self.log.logMessage(f'Repair Data: {len(repair_ids)} / {len(database)}')
        pbar = tqdm(total=len(repair_ids), desc="Repairing")

        # 3. Main Evaluation & Repair Loop
        for data in database:
            q_id = data.get("id")
            record = self.file.getRecordById(q_id)

            # Case A: The record is intact and has an answer. Just carry it over.
            if record and record.get("MyAnswer"):
                result.append(record)
                continue
            
            # Case B: Needs repair. 
            # Thanks to Dataset Double Loading, data["question"] is ALREADY translated!
            current_question = data.get("question", "")
            
            # Reuse the getPrompt method inherited from OnlyOneLanguage
            prompt = self.getPrompt(current_question)
            
            # Trigger the model for reasoning
            resultAnswer = self.model.getRes(prompt)
            my_answer = self.parseAnswer(resultAnswer)

            # Append the newly repaired record
            result.append({
                "id": q_id,
                "Question": current_question,
                "Result": resultAnswer,
                "Answer": data.get("answer", ""),
                "MyAnswer": my_answer
            })

            # Log the repaired interaction
            self.log.logMessage(f'當前問題 (Question)：\n{current_question}')
            self.log.logMessage(f'模型輸出 (Result)：\n{resultAnswer}')
            self.log.logMessage(f'My Answer: {my_answer} | Correct Answer: {data.get("answer", "")}\n')

            pbar.update()

        pbar.close()

        return result