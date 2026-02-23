from enum import Enum

class DatasetType(str, Enum):
    MATHQA = "mathqa"
    COMMONSENSEQA = "commonsenseqa"
    MGSM = "mgsm"
    MMLU = "mmlu"
    TRUTHFULQA = "truthfulqa"
    XCOPA = "xcopa"
    MLECQA = "mlecqa"
    CMBEXAM = "cmb"

class DatasetNameType(str, Enum):
    MATHQA = "MathQA"
    COMMONSENSEQA = "CommonSenseQA"
    MGSM = "MGSM"
    MMLU = "MMLU"
    TRUTHFULQA = "TruthfulQA"
    XCOPA = "XCOPA"
    MLECQA = "MlecQA"
    CMBEXAM = "CMB Exam"

# 直接用 value 取字串列表
DATASET_LIST = [d.value for d in DatasetType]

# key 改成字串，對應 dataset class
def get_dataset_map():
    # ← 只有真正用到時才 import，不會循環
    from Dataset.MathQA import MathQA as _MathQA
    from Dataset.CommonsenseQA import CommonsenseQA as _CommonsenseQA
    from Dataset.MGSM import MGSM as _MGSM
    from Dataset.MMLU import MMLU as _MMLU
    from Dataset.TruthfulQA import TruthfulQA as _TruthfulQA
    from Dataset.XCOPA import XCOPA as _XCOPA
    from Dataset.MLECQA import MLECQA as _MLECQA
    from Dataset.CMBExam import CMBExam as _CMBExam

    return {
        DatasetType.MATHQA.value: _MathQA,
        DatasetType.COMMONSENSEQA.value: _CommonsenseQA,
        DatasetType.MGSM.value: _MGSM,
        DatasetType.MMLU.value: _MMLU,
        DatasetType.TRUTHFULQA.value: _TruthfulQA,
        DatasetType.XCOPA.value: _XCOPA,
        DatasetType.MLECQA.value: _MLECQA,
        DatasetType.CMBEXAM.value: _CMBExam
    }

# key 改成字串，對應 dataset 名稱
DATASET_TO_NAME = {
    member: DatasetNameType[member.name].value for member in DatasetType
}