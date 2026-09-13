import numpy as np

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
    # 共用工具：對錯判定、split-half 切分、錨點選擇、recovery 系列公式
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

    @staticmethod
    def makeSplits(n: int, reps: int = 200, seed: int = 0) -> np.ndarray:
        """(reps, n) 的 bool 矩陣，True = H1（n // 2 題）、False = H2。同 n、同 seed 必得同一組切分。"""
        rng = np.random.default_rng(seed)
        return np.stack([rng.permutation(n) < n // 2 for _ in range(reps)])

    @staticmethod
    def pickMax(k1: int, k2: int, seed: int, rep: int, key1: int, key2: int) -> bool:
        """
        在 H1 上決定 L_max：k1 / k2 是兩者在 H1 的答對題數，回傳「第 1 個是否為 L_max」。
        平手時用 (seed, rep, 兩個 key) 決定的亂數，與參數順序無關，不同程式呼叫會得到同一個結果。
        """
        if k1 != k2:
            return k1 > k2
        lower_wins = np.random.default_rng([seed, rep, min(key1, key2), max(key1, key2)]).random() < 0.5
        return (key1 < key2) == lower_wins

    @staticmethod
    def recoveryStats(cA, cB, cf, dis, mask) -> dict:
        """
        paper_status 0A-1 的公式，只用 mask 內的題目（A = 錨點）:
            D = 兩個初答不同的題目；d = |D| / N
            n_A / n_B = D 中 A / B 正確的題數；c = (n_A + n_B) / |D|；m = c / 2；w_A = n_A / (n_A + n_B)
            recovery_blind = 2·w_A − 1
            recovery = (acc_agg − m) / (c − m)，acc_agg = D 中最終答案正確的比例
            skill = (recovery − recovery_blind) / (1 − recovery_blind)
        cA / cB: 初答是否正確；cf: 最終答案是否正確；dis: 兩個初答是否不同（皆為同長度的 bool 陣列）
        """
        nan = float("nan")
        D = dis & mask
        N, nD = int(mask.sum()), int(D.sum())
        nA, nB = int((D & cA).sum()), int((D & cB).sum())
        nR = nA + nB
        d = nD / N if N else nan
        c = nR / nD if nD else nan
        m = c / 2
        if nR == 0:
            return dict(N=N, D=nD, d=d, c=c, m=m, n_A=nA, n_B=nB,
                        w_A=nan, recovery_blind=nan, recovery=nan, skill=nan)
        w_A = nA / nR
        recovery_blind = 2 * w_A - 1
        recovery = (int((D & cf).sum()) / nD - m) / (c - m)
        skill = (recovery - recovery_blind) / (1 - recovery_blind) if recovery_blind < 1 else nan
        return dict(N=N, D=nD, d=d, c=c, m=m, n_A=nA, n_B=nB, w_A=w_A,
                    recovery_blind=recovery_blind, recovery=recovery, skill=skill)
