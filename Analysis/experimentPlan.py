from dataclasses import dataclass
from itertools import combinations

from Arm.ArmSpec import ArmSpec
from Arm.AxisType import AxisType, ANCHOR_ARM_ID
from Strategy.StrategyType import LANGUAGE_STR_LIST

# ------------------------------------------------------------------
# K=2 實驗計畫（todo.png）：14 條 path 與 19 個配對
# ------------------------------------------------------------------

# path 代號 -> arm_id
PATHS = {
    "EN": "L:en", "ZH": "L:zh", "JA": "L:ja", "RU": "L:ru", "ES": "L:es",
    "S1": "S:T1.0:seed1", "S2": "S:T1.0:seed2",
    "P1": "P:expert", "P2": "P:skeptic",
    "W1": "W:rewrite1", "W2": "W:rewrite2",
    "R": "R:short_cot",
    "SR-EN": "F:en", "SR-ZH": "F:zh",
}
ARM_TO_PATH = {arm_id: path for path, arm_id in PATHS.items()}

# 檔名 stem -> arm_id（明確對照，不從 stem 反推：S_T1.0_seed1 不能用字串替換還原）
STEM_TO_ARM = {ArmSpec.from_arm_id(arm_id).file_stem: arm_id for arm_id in PATHS.values()}

# pickMax 平手判定的 key：L arm 用 LANGUAGE_STR_LIST 的 index（與 0A-1 相同），
# 其餘 path 依 PATHS 順序接在後面（5, 6, ...），每個 key 都不同
_L_ARMS = [arm_id for arm_id in PATHS.values() if ArmSpec.from_arm_id(arm_id).axis == AxisType.LANGUAGE]
_OTHER_ARMS = [arm_id for arm_id in PATHS.values() if arm_id not in _L_ARMS]
TIE_BREAK_KEY = {arm_id: LANGUAGE_STR_LIST.index(ArmSpec.from_arm_id(arm_id).language) for arm_id in _L_ARMS}
TIE_BREAK_KEY.update({arm_id: len(LANGUAGE_STR_LIST) + i for i, arm_id in enumerate(_OTHER_ARMS)})


@dataclass(frozen=True)
class Pair:
    """一個 K=2 候選配對；arm_a / arm_b 的順序 = 聚合檔檔名中的順序。"""
    arm_a: str
    arm_b: str

    @property
    def arms(self) -> list[str]:
        return [self.arm_a, self.arm_b]

    @property
    def label(self) -> str:
        return f"{ARM_TO_PATH[self.arm_a]}+{ARM_TO_PATH[self.arm_b]}"

    @property
    def axis(self) -> str:
        """配對代表的多樣性來源：非 L 的那一軸；兩個都是 L 時為 L。"""
        axes = [ArmSpec.from_arm_id(arm_id).axis for arm_id in self.arms]
        others = [axis for axis in axes if axis != AxisType.LANGUAGE]
        return others[0] if others else AxisType.LANGUAGE.value

    @property
    def symmetric(self) -> bool:
        """todo.png 的「對稱」：兩個候選來自同一軸且都不是錨點 L:en（S1+S2 對稱、EN+S1 不對稱）。"""
        axes = {ArmSpec.from_arm_id(arm_id).axis for arm_id in self.arms}
        return len(axes) == 1 and ANCHOR_ARM_ID not in self.arms

    @property
    def n_english(self) -> int:
        """todo.png 的「英文數」：候選中用英文作答的個數。"""
        return sum(ArmSpec.from_arm_id(arm_id).lang == "en" for arm_id in self.arms)


# 10 個語言配對（順序與 legacy debate 檔相同）+ 9 個其他軸的配對
_LANGUAGE_PATHS = ["EN", "ZH", "JA", "RU", "ES"]
PAIRS = [Pair(PATHS[a], PATHS[b]) for a, b in combinations(_LANGUAGE_PATHS, 2)] + [
    Pair(PATHS[a], PATHS[b]) for a, b in [
        ("EN", "S1"), ("S1", "S2"),
        ("EN", "P1"), ("P1", "P2"),
        ("EN", "W1"), ("W1", "W2"),
        ("EN", "R"),
        ("EN", "SR-EN"), ("ZH", "SR-ZH"),
    ]
]
ARMS_TO_PAIR = {frozenset(pair.arms): pair for pair in PAIRS}

# 以聚合檔分析的聚合器；Blind 不跑聚合器，由分析程式計算
FILE_AGGREGATORS = ("judge", "debate")
