import numpy as np

from File.File import File
from Log.Log import Log
from Strategy.StrategyType import StrategyType, LANGUAGE_STR_LIST
from Test.Test import Test

# 在 H2 上計算、再對 reps 次切分取平均的量（metadata 中加 _H2 字尾）
H2_KEYS = ("n_A", "n_B", "w_A", "recovery_blind", "recovery", "skill")


class TestRecoveryBlind(Test):
    """
    paper_status 0A-1：對每個 challenge（兩個 agent 辯論）結果檔計算 recovery_blind 等量，
    寫回 metadata["RecoveryBlind"]。

    題目範圍:
        d, c, m                                          全部題目（不涉及選擇）
        錨點 A = 兩個語言中單語準確率較高者                  在 H1 決定（Test.makeSplits / Test.pickMax）
        n_A, n_B, w_A, recovery_blind, recovery, skill     在同一個 H2 上計算，再對 reps 次取平均
    recovery 必須和 recovery_blind 用同一批 H2 分歧題：recovery 的分母 c − m 與 recovery_blind 的分子共用這些題目。
    切分與錨點規則和 split_half_gap.py（0A-4）完全相同。

    初答取 AnswerRecord1[0] / AnswerRecord2[0]（Challenge 直接沿用兩個 baseline 檔的 MyAnswer）。
    """
    def __init__(self, seed: int = 0, reps: int = 200):
        super().__init__()
        self.name: str = "Test Recovery Blind"
        self.seed = seed
        self.reps = reps

    @staticmethod
    def compute(c1, c2, cf, dis, name1: str, name2: str, key1: int, key2: int,
                seed: int = 0, reps: int = 200) -> dict:
        """
        c1 / c2: agent1 / agent2 初答是否正確；cf: 最終答案是否正確；dis: 兩個初答是否不同
        （皆為依題目排序的 bool 陣列）。name1 / name2 是錨點的顯示名稱，key1 / key2 用於平手判定。
        test_em_legacy.py 也呼叫這個函式。
        """
        c1, c2, cf, dis = (np.asarray(x, dtype=bool) for x in (c1, c2, cf, dis))
        N = len(c1)
        full = Test.recoveryStats(c1, c2, cf, dis, np.ones(N, dtype=bool))

        stats_H2, first_picked = [], []
        for rep, h1 in enumerate(Test.makeSplits(N, reps, seed)):
            first = Test.pickMax(int(c1[h1].sum()), int(c2[h1].sum()), seed, rep, key1, key2)  # H1 選錨點
            cA, cB = (c1, c2) if first else (c2, c1)
            stats_H2.append(Test.recoveryStats(cA, cB, cf, dis, ~h1))                          # H2 計算
            first_picked.append(first)

        first_rate = float(np.mean(first_picked))
        result = {
            "seed": seed,
            "reps": reps,
            "N": full["N"],
            "D": full["D"],
            "d": full["d"],
            "c": full["c"],
            "m": full["m"],
            "anchor": name1 if first_rate >= 0.5 else name2,
            "anchor_rate": max(first_rate, 1 - first_rate),
        }
        for key in H2_KEYS:
            values = np.array([s[key] for s in stats_H2], dtype=float)
            result[f"{key}_H2"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        # H2 中沒有可救分歧題（recovery 無定義）/ recovery_blind = 1（skill 無定義）的次數
        result["reps_undefined"] = int(sum(np.isnan(s["recovery"]) for s in stats_H2))
        result["reps_skill_undefined"] = int(sum(np.isnan(s["skill"]) for s in stats_H2))
        return result

    def runTest(self, fileList: list[File], log: Log):
        for file in fileList:
            if file.getStrategyConfig().strategyType != StrategyType.CHALLENGE:
                print(f"[TestRecoveryBlind] 略過非 challenge 檔: {file.file_path}")
                continue

            records = [file.records_map[q_id] for q_id in sorted(file.records_map)]
            if not records or any(not r.get("AnswerRecord1") or not r.get("AnswerRecord2") for r in records):
                print(f"[TestRecoveryBlind] 略過缺 AnswerRecord 的檔案: {file.file_path}")
                continue
            log.logInfo(file)

            DatasetClass = self.getDatasetClass(file)
            init1 = [str(r["AnswerRecord1"][0]) for r in records]
            init2 = [str(r["AnswerRecord2"][0]) for r in records]
            gold = [str(r.get("Answer", "")) for r in records]
            final = [str(r.get("MyAnswer", "")) for r in records]

            c1 = [DatasetClass.compareTwoAnswer(g, a) for g, a in zip(gold, init1)]
            c2 = [DatasetClass.compareTwoAnswer(g, a) for g, a in zip(gold, init2)]
            cf = [DatasetClass.compareTwoAnswer(g, a) for g, a in zip(gold, final)]
            dis = [not DatasetClass.compareTwoAnswer(a, b) for a, b in zip(init1, init2)]

            lang1, lang2 = file.getLanguage()
            result = self.compute(c1, c2, cf, dis, lang1, lang2,
                                  LANGUAGE_STR_LIST.index(lang1), LANGUAGE_STR_LIST.index(lang2),
                                  self.seed, self.reps)

            log.logMessage(f'[{file.getModelConfig().modelName} on {file.getDatasetConfig().displayName} '
                           f'({lang1} vs {lang2})]\n'
                           f'anchor: {result["anchor"]} ({result["anchor_rate"]:.0%})  '
                           f'recovery_blind_H2: {result["recovery_blind_H2"]:.4f}  '
                           f'recovery_H2: {result["recovery_H2"]:.4f}  '
                           f'skill_H2: {result["skill_H2"]:.4f}')

            file.updateMetadata("RecoveryBlind", result)
            file.save()
