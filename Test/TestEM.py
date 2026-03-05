from File.File import File
from Log.Log import Log
from Dataset.DatasetType import get_dataset_map, DatasetType
from Test.Test import Test

class TestEM(Test):
    """
    Evaluates the Exact Match (EM) accuracy of the model's answers.
    Iterates through the records, uses the dataset's specific comparison logic, 
    and automatically updates the JSON file's metadata with the final score.
    """
    def __init__(self):
        super().__init__()
        self.name: str = "Test Exact Match"

    def runTest(self, fileList: list[File], log: Log):
        """
        Executes the exact match test across a list of parsed ResultFile objects.
        
        Args:
            fileList (list[File]): A list of File objects to evaluate.
            log (Log): The logging utility for outputting results.
        """
        for file in fileList:
            # 1. Provide basic info to the log
            # Depending on your log implementation, you might want to pass config objects instead
            log.logInfo(file)
            
            # 2. Dynamically instantiate the correct Dataset Class to use its specific compareTwoAnswer method
            # We extract the datasetType string from the Config and convert it to the Enum
            dataset_type_str = file.getDatasetConfig().datasetType
            dataset_enum = DatasetType(dataset_type_str)
            DatasetClass = get_dataset_map()[dataset_enum]
            
            correct_cnt = 0
            total_cnt = 0

            # 3. Iterate through the robust O(1) records_map instead of a fragile list
            for q_id, record in file.records_map.items():
                total_cnt += 1
                
                # Safely get the answers, defaulting to empty strings if missing
                ans = str(record.get("Answer", ""))
                my_ans = str(record.get("MyAnswer", ""))
                
                # Use the dataset's static method to compare answers (e.g., handles math float equality)
                if DatasetClass.compareTwoAnswer(ans, my_ans):
                    correct_cnt += 1
            
            # 4. Calculate performance securely (prevent division by zero)
            accuracy = correct_cnt / total_cnt if total_cnt > 0 else 0.0
            accuracy_percentage = accuracy * 100
            
            # 5. Log the performance to the terminal
            log.logMessage(f'[{file.getModelConfig().modelName} on {file.getDatasetConfig().displayName}]')
            log.logMessage(f'Performance: {correct_cnt} / {total_cnt} ({accuracy_percentage:.2f}%)\n')

            # 6. 🌟 The Magic Step: Write the metrics back to the JSON Metadata!
            file.updateMetadata("ExactMatch_Correct", correct_cnt)
            file.updateMetadata("ExactMatch_Total", total_cnt)
            file.updateMetadata("ExactMatch_Accuracy", round(accuracy, 4))
            
            # Save the updated changes back to the physical JSON file
            file.save()