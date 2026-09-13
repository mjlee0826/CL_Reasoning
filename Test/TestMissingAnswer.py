from File.File import File
from Log.Log import Log
from Test.Test import Test


def isMissing(value) -> bool:
    """與 check_results.py 相同的判定：缺欄位、空值、"null"、"None"。"""
    return not value or str(value).strip() in ("", "null", "None")


class TestMissingAnswer(Test):
    """
    paper_status 0A-3：統計每個結果檔沒有解析出答案的題數，寫回 metadata["MissingAnswer"]。
        n               題數
        missing_final   MyAnswer 缺失的題數
        rate_final      missing_final / n
    challenge 檔（紀錄帶 AnswerRecord1 / AnswerRecord2）另外記錄兩個 agent 初答的缺失數:
        missing_init1   AnswerRecord1[0] 缺失的題數
        missing_init2   AnswerRecord2[0] 缺失的題數
    """
    def __init__(self):
        super().__init__()
        self.name: str = "Test Missing Answer"

    def runTest(self, fileList: list[File], log: Log):
        for file in fileList:
            log.logInfo(file)
            records = list(file.records_map.values())
            n = len(records)
            missing_final = sum(isMissing(r.get("MyAnswer")) for r in records)
            summary = {
                "n": n,
                "missing_final": missing_final,
                "rate_final": missing_final / n if n else 0.0,
            }

            if records and all("AnswerRecord1" in r and "AnswerRecord2" in r for r in records):
                summary["missing_init1"] = sum(isMissing((r.get("AnswerRecord1") or [None])[0]) for r in records)
                summary["missing_init2"] = sum(isMissing((r.get("AnswerRecord2") or [None])[0]) for r in records)

            log.logMessage(f'[{file.file_path}] {summary}')
            file.updateMetadata("MissingAnswer", summary)
            file.save()
