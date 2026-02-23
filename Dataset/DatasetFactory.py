from Dataset.Dataset import Dataset
from Dataset.MathQA import MathQA
from Dataset.CommonsenseQA import CommonsenseQA
from Dataset.MGSM import MGSM
from Dataset.MMLU import MMLU
from Dataset.TruthfulQA import TruthfulQA
from Dataset.XCOPA import XCOPA
from Dataset.MLECQA import MLECQA
from Dataset.CMBExam import CMBExam

from Dataset.DatasetType import DatasetType

class DatasetFactory():
    def __init__(self):
        pass

    def buildDataset(self, type, *args, **kwargs) -> Dataset:
        if type == DatasetType.MATHQA:
            return MathQA(*args, **kwargs)
        elif type == DatasetType.COMMONSENSEQA:
            return CommonsenseQA(*args, **kwargs)
        elif type == DatasetType.MGSM:
            return MGSM(*args, **kwargs)
        elif type == DatasetType.MMLU:
            return MMLU(*args, **kwargs)
        elif type == DatasetType.TRUTHFULQA:
            return TruthfulQA(*args, **kwargs)
        elif type == DatasetType.XCOPA:
            return XCOPA(*args, **kwargs)
        elif type == DatasetType.MLECQA:
            return MLECQA(*args, **kwargs)
        elif type == DatasetType.CMBEXAM:
            return CMBExam(*args, **kwargs)
        else:
            print('Dataset doesn\'t exist!')
            return None
