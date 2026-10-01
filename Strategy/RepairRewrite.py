from Model.Model import Model
from Model.ModelFactory import ModelFactory
from Model.ModelType import ModelType
from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DatasetType
from Strategy.StrategyConfig import StrategyConfig
from Strategy.Rewrite import Rewrite
from Log.Log import Log
from File.File import File

from tqdm import tqdm


class RepairRewrite(Rewrite):
    """
    Repair pass for Experiment 1 output (mirrors Strategy.RepairOnlyOneLanguage).

    Rebuilds the Model / Dataset straight from the result file's own metadata, scans for
    records whose 'Rewritten' field is missing / empty / an API error message,
    re-runs only those, and overwrites the file in place.

    Only for version-1 files written by the old runner. Version >= 2 files need the earlier versions in
    their prompt and never store failed calls: rerun run_rewrite.py --version n to fill them.
    """
    def __init__(self, config: StrategyConfig, log: Log, file: File):
        if file.getStrategyConfig().rewriteVersion >= 2:
            raise ValueError(f"{file.file_path} is a version >= 2 rewrite; rerun run_rewrite.py with that --version instead")

        model_config = file.getModelConfig()
        self.model: Model = ModelFactory().buildModel(ModelType(model_config.modelType), model_config)

        dataset_config = file.getDatasetConfig()
        self.dataset: Dataset = DatasetFactory().buildDataset(DatasetType(dataset_config.datasetType), dataset_config)

        super().__init__(config, self.model, self.dataset, log)
        self.file: File = file

    def checkError(self, record: dict) -> bool:
        if not record:
            return True

        rewritten = record.get("Rewritten")
        if rewritten is None or str(rewritten).strip() == "":
            return True

        # QWEN returns "Error Code: ...", the other models "Error in <model> model: ..."
        if "Error Code" in str(rewritten) or str(rewritten).lstrip().startswith("Error"):
            return True

        return False

    def getRes(self) -> list:
        self.log.logInfo(self, self.model, self.dataset)
        self.log.logMessage("Repair Mode (Rewrite): only re-running records with missing/error 'Rewritten'.")

        database = self.dataset.getData()
        repair_ids = []

        for data in database:
            q_id = data.get("id")
            record = self.file.getRecordById(q_id)
            if self.checkError(record):
                repair_ids.append(q_id)

        self.log.logMessage(f'Repair Data: {len(repair_ids)} / {len(database)}')

        if not repair_ids:
            self.log.logMessage("All rewrite records are intact! No repair needed.")
            return []

        pbar = tqdm(total=len(repair_ids), desc="Repairing Rewrite")

        db_map = {data.get("id"): data for data in database}
        for q_id in repair_ids:
            record = self.file.getRecordById(q_id)
            current_question = record.get("Question", "") if record else db_map.get(q_id, {}).get("question", "")

            rewritten_question = self.model.getRes(self.getPrompt(current_question))

            self.file.updateRecord(q_id, {
                "id": q_id,
                "Question": current_question,
                "Rewritten": rewritten_question
            })

            self.log.logMessage(f'改寫問題 (Rewritten)：\n{rewritten_question}')
            pbar.update()

        pbar.close()
        self.file.save()

        return repair_ids
