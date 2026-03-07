from Model.Model import Model
from Model.ModelFactory import ModelFactory
from Model.ModelType import ModelType
from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType
from Strategy.StrategyConfig import StrategyConfig
from Strategy.Challenge import Challenge
from Log.Log import Log
from File.File import File

from tqdm import tqdm

class RepairChallenge(Challenge):
    """
    Repair strategy inheriting from Challenge.
    It scans a previously generated Challenge JSON result file, identifies missing or erroneous 
    debate records, and re-triggers the debate/judge process to repair them.
    """
    def __init__(self, config: StrategyConfig, log: Log, file: File):
        self.file: File = file
        
        # 1. Safely retrieve and instantiate the Model using Enum type casting
        model_config = file.getModelConfig()
        self.model: Model = ModelFactory().buildModel(ModelType(model_config.modelType), model_config)

        # 2. Safely retrieve and instantiate the Dataset using Enum type casting
        dataset_config = file.getDatasetConfig()
        self.dataset: Dataset = DatasetFactory().buildDataset(DatasetType(dataset_config.datasetType), dataset_config)

        # 3. Retrieve the underlying File1 and File2 paths from metadata to reconstruct the agents
        file1_path = file.metadata.get("File1", {}).get("path")
        file2_path = file.metadata.get("File2", {}).get("path")
        
        if not file1_path or not file2_path:
            raise ValueError("❌ RepairChallenge requires valid File1 and File2 paths in the metadata.")
            
        file1 = File(file1_path)
        file2 = File(file2_path)

        # 4. Initialize the parent Challenge class
        super().__init__(config, self.model, self.dataset, log, file1, file2)
    
    def checkError(self, record: dict) -> bool:
        """
        Evaluates whether a Challenge record needs to be repaired.
        Checks for missing answers or API Error Codes within the debate histories.
        """
        if not record:
            return True
        
        # The extracted final answer is missing or empty
        my_answer = record.get('MyAnswer')
        if my_answer is None or my_answer == '':
            return True
            
        # Check if the Judge phase threw an API Error
        if record.get('Result3') and 'Error Code' in record.get('Result3'):
            return True
            
        # Check if any Agent's response during the debate phase threw an API Error
        for r in record.get('Record1', []):
            if r.get('role') == 'assistant' and 'Error Code' in r.get('content', ''):
                return True
                
        for r in record.get('Record2', []):
            if r.get('role') == 'assistant' and 'Error Code' in r.get('content', ''):
                return True
            
        for r in record.get('AnswerRecord1', []):
            if r == '':
                return True
                
        for r in record.get('AnswerRecord2', []):
            if r == '':
                return True
        
        return False

    def getRes(self) -> list:
        """
        Executes the repair pipeline. Identifies broken debate records, re-evaluates them 
        by simulating the debate and judge phases again, and saves the updated data.
        """
        self.log.logInfo(self, self.model, self.dataset, self.file1, self.file2)
        self.log.logMessage("🔧 Repair Mode (Challenge): Re-running debates for missing or error records.")

        database = self.dataset.getData()
        repair_ids = []
        
        # Build an O(1) lookup map for the database to prevent loop variable contamination
        db_map = {data.get("id"): data for data in database}
        
        # Scan for IDs that require repairing
        for data in database:
            q_id = data.get("id")
            record = self.file.getRecordById(q_id)
            
            if self.checkError(record):
                repair_ids.append(q_id)

        self.log.logMessage(f'Repair Data: {len(repair_ids)} / {len(database)}')
        
        if not repair_ids:
            self.log.logMessage("🎉 All Challenge data is intact! No repair needed.")
            return []

        pbar = tqdm(total=len(repair_ids), desc="Repairing Challenge")

        for q_id in repair_ids:
            data = db_map.get(q_id)
            if not data:
                pbar.update()
                continue
                
            # To repair a Challenge, we MUST have the original baseline records
            rec1 = self.file1.getRecordById(q_id)
            rec2 = self.file2.getRecordById(q_id)
            
            if not rec1 or not rec2:
                self.log.logMessage(f"⚠️ Cannot repair ID {q_id}: Missing baseline record in File1 or File2.")
                pbar.update()
                continue

            question1, question2 = rec1.get("Question", ""), rec2.get("Question", "")
            result1, result2 = rec1.get("Result", ""), rec2.get("Result", "")
            answer1, answer2 = rec1.get("MyAnswer", ""), rec2.get("MyAnswer", "")
            correct_answer = data.get("answer", "")

            # Re-initialize variables for the repair run
            cur_turn = 0
            resultOutput3 = ""
            myAnswer = ""
            record1 = [
                {"role": "user", "content": question1},
                {"role": "assistant", "content": result1}
            ]
            record2 = [
                {"role": "user", "content": question2},
                {"role": "assistant", "content": result2}
            ]
            answerRecord1, answerRecord2 = [answer1], [answer2]

            self.log.logMessage(f'\n【Repairing Challenge ID: {q_id}】')

            # --- Re-run the Challenge Logic ---
            if self.dataset.compareTwoAnswer(answer1, answer2):
                myAnswer = answer1
                self.log.logMessage(f'Result: Initial agreement. No debate needed.')
            else:
                record1, record2, result1, result2, answer1, answer2, answerRecord1, answerRecord2, cur_turn = \
                    self.runChallenge(question1, question2, result1, result2, answer1, answer2) 

                if self.dataset.compareTwoAnswer(answer1, answer2):
                    myAnswer = answer1
                    self.log.logMessage(f'Result: Agents reached consensus after {cur_turn} turns!')
                else:
                    judge_lang = self.lang1
                    resultOutput3 = self.model.getRes(self.getJudgePrompt(judge_lang, question1, result1, result2))
                    myAnswer = self.parseAnswer(resultOutput3)
                    self.log.logMessage(f'Result 3 (Judge Invoked): \n{resultOutput3}')

            self.log.logMessage(f'My Answer: {myAnswer} | Correct Answer: {correct_answer}\n')

            # Overwrite the repaired record in the File's memory map
            self.file.updateRecord(q_id, {
                "id": q_id,
                "Question1": question1,
                "Question2": question2,
                "Record1": record1,
                "Record2": record2,
                "AnswerRecord1": answerRecord1,
                "AnswerRecord2": answerRecord2,
                "Times": cur_turn,
                "Result3": resultOutput3,
                "Answer": correct_answer,
                "MyAnswer": myAnswer
            })

            pbar.update()

        pbar.close()
        
        # Automatically save the repaired memory map back to the physical JSON file
        self.file.save()

        return repair_ids