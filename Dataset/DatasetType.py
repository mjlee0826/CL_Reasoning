from enum import Enum

# Define the core identifiers for each dataset used internally across the system.
# Inheriting from (str, Enum) allows these to be compared directly with strings if needed,
# while maintaining strict type safety.
class DatasetType(str, Enum):
    MATHQA = "mathqa"
    COMMONSENSEQA = "commonsenseqa"
    MGSM = "mgsm"
    MMLU = "mmlu"
    TRUTHFULQA = "truthfulqa"
    XCOPA = "xcopa"
    MLECQA = "mlecqa"
    CMBEXAM = "cmb"

# Define the formatted, human-readable display names for each dataset.
# Useful for logging, generating reports, or UI rendering.
class DatasetDisplayNameType(str, Enum):
    MATHQA = "MathQA"
    COMMONSENSEQA = "CommonSenseQA"
    MGSM = "MGSM"
    MMLU = "MMLU"
    TRUTHFULQA = "TruthfulQA"
    XCOPA = "XCOPA"
    MLECQA = "MlecQA"
    CMBEXAM = "CMB Exam"

# A list containing the raw string values of all supported datasets.
# Useful for input validation (e.g., checking if a command-line argument is valid).
DATASET_STR_LIST = [d.value for d in DatasetType]

def get_dataset_map():
    """
    Returns a dictionary mapping DatasetType enums to their respective class definitions.
    
    Uses "lazy importing" (importing modules inside the function rather than at the top of the file)
    to prevent circular import errors. This ensures classes are only loaded into memory 
    when the factory actually needs to instantiate them.
    """
    # Lazy import of concrete dataset implementation classes
    from Dataset.MathQA import MathQA as _MathQA
    from Dataset.CommonsenseQA import CommonsenseQA as _CommonsenseQA
    from Dataset.MGSM import MGSM as _MGSM
    from Dataset.MMLU import MMLU as _MMLU
    from Dataset.TruthfulQA import TruthfulQA as _TruthfulQA
    from Dataset.XCOPA import XCOPA as _XCOPA
    from Dataset.MLECQA import MLECQA as _MLECQA
    from Dataset.CMBExam import CMBExam as _CMBExam

    # Mapping DatasetType (Enum objects) directly to the loaded classes
    return {
        DatasetType.MATHQA: _MathQA,
        DatasetType.COMMONSENSEQA: _CommonsenseQA,
        DatasetType.MGSM: _MGSM,
        DatasetType.MMLU: _MMLU,
        DatasetType.TRUTHFULQA: _TruthfulQA,
        DatasetType.XCOPA: _XCOPA,
        DatasetType.MLECQA: _MLECQA,
        DatasetType.CMBEXAM: _CMBExam
    }

# Dictionary mapping the internal DatasetType to its corresponding DatasetDisplayNameType.
# Note: The values here remain Enum objects (not pure strings) to preserve 
# type safety and object semantics throughout the core logic.
DATASET_TO_DISPLAYNAME = {
    member: DatasetDisplayNameType[member.name] for member in DatasetType
}