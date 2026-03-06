from Model.Model import Model
from Model.ModelFactory import ModelFactory
from Model.ModelConfig import ModelConfig
from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetConfig import DatasetConfig
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Strategy.OnlyOneLanguage import OnlyOneLanguage
from Log.Log import Log
from File.File import File

from tqdm import tqdm

class RepairOnlyOneLanguage(OnlyOneLanguage):
    def __init__(self, config: StrategyConfig, log: Log, file: File):
        # Initialize the parent class (OnlyOneLanguage) to inherit prompt and logic settings
        model_config = file.getModelConfig()
        self.model: Model = ModelFactory().buildModel(model_config.modelType, model_config)

        dataset_config = file.getDatasetConfig()
        self.dataset:Dataset = DatasetFactory().buildDataset(dataset_config.datasetType, dataset_config)

        super().__init__(config, self.model, self.dataset, log)
        self.file: File = file
    
    def checkError(self, record: dict):
        if not record:
            return True
        
        if record.get('Translated') and 'Error Code' in record.get('Translated'):
            return True
        
        if record.get('Result') and 'Error Code' in record.get('Result'):
            return True
        
        if record.get('MyAnswer') != None and record.get('MyAnswer') == '':
            return True
        
        return False

    def getRes(self) -> list:
        self.log.logInfo(self, self.model, self.dataset)
        self.log.logMessage("Repair Mode: Only processing data with missing or empty 'MyAnswer'.")

        database = self.dataset.getData()
        repair_ids = []
        
        for data in database:
            q_id = data.get("id")
            record = self.file.getRecordById(q_id)
            
            if self.checkError(record):
                repair_ids.append(q_id)

        self.log.logMessage(f'Repair Data: {len(repair_ids)} / {len(database)}')
        pbar = tqdm(total=len(repair_ids), desc="Repairing")

        for q_id in repair_ids:
            record = self.file.getRecordById(q_id)
            
            current_question = record.get("Question", "")
            prompt = self.getPrompt(current_question)
            
            # Trigger the model for reasoning
            resultAnswer = self.model.getRes(prompt)
            my_answer = self.parseAnswer(resultAnswer)

            # Append the newly repaired record
            self.file.updateRecord(q_id, {
                "id": q_id,
                "Question": current_question,
                "Result": resultAnswer,
                "Answer": record.get("Answer", ""),
                "MyAnswer": my_answer
            })

            # Log the repaired interaction
            self.log.logMessage(f'當前問題 (Question)：\n{current_question}')
            self.log.logMessage(f'模型輸出 (Result)：\n{resultAnswer}')
            self.log.logMessage(f'My Answer: {my_answer} | Correct Answer: {record.get("Answer", "")}\n')

            pbar.update()

        pbar.close()
        self.file.save()

        return repair_ids