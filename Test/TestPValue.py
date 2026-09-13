from File.File import File
from Log.Log import Log
from scipy import stats

from Test.Test import Test

class TestPValue(Test):
    """
    兩個結果檔的配對 t 檢定：逐題 0/1（是否答對），以題目 id 配對。
    需要剛好兩個檔案：python test_em.py -t testp --testfile a.json b.json
    """
    def __init__(self):
        super().__init__()
        self.name: str = "Test p Value"

    def runTest(self, fileList: list[File], log: Log):
        fileList = list(fileList)
        if len(fileList) != 2:
            print(f"[TestPValue] 需要剛好兩個檔案，收到 {len(fileList)} 個")
            return

        file1, file2 = fileList
        log.logInfo(file1)
        log.logInfo(file2)

        # 以題目 id 配對（舊版呼叫的 getData / getDatasetName / getDataNums 在 File 中已不存在）
        correct1, correct2 = self.getCorrectMap(file1), self.getCorrectMap(file2)
        ids = sorted(correct1.keys() & correct2.keys())
        if not ids:
            print("[TestPValue] 兩個檔案沒有共同的題目 id")
            return
        score1 = [int(correct1[q_id]) for q_id in ids]
        score2 = [int(correct2[q_id]) for q_id in ids]

        t_stat, p_value = stats.ttest_rel(score1, score2)
        # 直接 print：預設的 NoLog 不會輸出 logMessage
        print(f'{file1.file_path} vs {file2.file_path}')
        print(f'paired n: {len(ids)}  accuracy: {sum(score1) / len(ids):.4f} vs {sum(score2) / len(ids):.4f}')
        print(f'p value: {p_value}\nt stat: {t_stat}')
