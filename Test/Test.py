from Dataset.DatasetType import get_dataset_map, DatasetType
from File.File import File
from Log.Log import Log

class Test():
    def __init__(self):
        self.name: str = ""
    
    def printName(self):
        print(f'Test {self.name}')
    
    def getName(self) -> str:
        return self.name

    def runTest(self, fileList: list[File], log: Log):
        return

    # ------------------------------------------------------------------
    # 共用工具：legacy 結果檔的對錯判定（split-half / recovery 公式在 Analysis/splitHalf.py）
    # TestEM / TestPValue / TestRecoveryBlind / split_half_gap.py 共用，確保定義一致
    # ------------------------------------------------------------------
    @staticmethod
    def getDatasetClass(file: File):
        """依 metadata 的 datasetType 取得對應的 Dataset class（使用它的 compareTwoAnswer）。"""
        return get_dataset_map()[DatasetType(file.getDatasetConfig().datasetType)]

    @staticmethod
    def getCorrectMap(file: File) -> dict:
        """{id: MyAnswer 是否正確}，統一用 Dataset.compareTwoAnswer 判定。"""
        DatasetClass = Test.getDatasetClass(file)
        return {
            q_id: DatasetClass.compareTwoAnswer(str(record.get("Answer", "")), str(record.get("MyAnswer", "")))
            for q_id, record in file.records_map.items()
        }
