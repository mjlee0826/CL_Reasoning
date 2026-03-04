from dataclasses import dataclass, fields, asdict, field
from Strategy.StrategyType import LanguageType

@dataclass
class DatasetConfig:
    """
    Data container for dataset configurations.
    """
    datasetType: str = ''
    displayName: str = ''
    nums: int = -1
    sample: int = 1

    language: LanguageType = LanguageType.ENGLISH.value
    
    # Use field(init=False) so this attribute is not expected in the __init__ arguments.
    # It will be calculated dynamically after initialization.
    dataNums: int = field(init=False)

    def __post_init__(self):
        """
        Built-in dataclass hook that runs immediately after __init__.
        Used here to calculate 'dataNums' based on the provided 'nums' and 'sample'.
        """
        self.dataNums = self.nums * self.sample

    @classmethod
    def from_args(cls, args):
        """
        Factory method to create a DatasetConfig instance from argparse results.
        Filters the arguments to prevent TypeError from unexpected keywords.
        """
        # Extract valid field names, but ONLY those allowed in __init__
        # (This prevents 'dataNums' from being incorrectly passed to the constructor)
        valid_keys = {f.name for f in fields(cls) if f.init}
        
        # Construct a dictionary by filtering the vars(args) dictionary,
        # skipping None values to preserve default dataclass values.
        filtered_data = {
            key: value 
            for key, value in vars(args).items() 
            if key in valid_keys and value is not None
        }

        # Unpack the filtered dictionary to instantiate the class
        return cls(**filtered_data)

    @classmethod
    def from_dict(cls, data_dict: dict):
        """
        從 JSON 字典建立 Config 實例。
        會自動過濾掉 data_dict 中不屬於此 dataclass 的 key，避免 TypeError。
        """
        # 取得這個 dataclass 允許在 __init__ 中初始化的所有欄位名稱
        valid_keys = {f.name for f in fields(cls) if f.init}
        
        # 過濾傳入的字典，只保留合法的 key
        filtered_data = {
            key: value 
            for key, value in data_dict.items() 
            if key in valid_keys
        }
        return cls(**filtered_data)
    
    def to_dict(self) -> dict:
        """
        Convert the dataclass instance to a standard dictionary.
        This will include the calculated 'dataNums' field as well.
        """

        self.dataNums = self.nums * self.sample
        return asdict(self)