from File.File import File
from Log.Log import Log
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
            
            # 2. Per-question correctness via the shared Test.getCorrectMap
            #    (uses the dataset's static compareTwoAnswer, e.g., handles math float equality)
            # 3. Count correct answers over the O(1) records_map
            correct_map = self.getCorrectMap(file)
            correct_cnt = sum(correct_map.values())
            total_cnt = len(correct_map)
            
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