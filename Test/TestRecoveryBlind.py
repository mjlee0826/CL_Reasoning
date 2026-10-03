from File.File import File
from Log.Log import Log
from Strategy.StrategyType import StrategyType, LANGUAGE_STR_LIST
from Test.Test import Test
from Analysis.splitHalf import splitHalfStats


class TestRecoveryBlind(Test):
    """
    paper_status 0A-1：對每個 challenge（兩個 agent 辯論）結果檔計算 recovery_blind 等量，
    寫回 metadata["RecoveryBlind"]。

    計算本身在 Analysis.splitHalf.splitHalfStats（錨點 A = 兩個語言中單語準確率較高者，在 H1 決定；
    n_A, n_B, w_A, recovery_blind, recovery, skill 在 H2 計算後對 reps 次取平均）。
    切分與錨點規則和 split_half_gap.py（0A-4）、run_analysis.py 完全相同。

    初答取 AnswerRecord1[0] / AnswerRecord2[0]（Challenge 直接沿用兩個 baseline 檔的 MyAnswer）。
    """
    def __init__(self, seed: int = 0, reps: int = 200):
        super().__init__()
        self.name: str = "Test Recovery Blind"
        self.seed = seed
        self.reps = reps

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
            result = splitHalfStats(c1, c2, cf, dis, lang1, lang2,
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
