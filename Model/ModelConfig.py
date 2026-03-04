from dataclasses import dataclass, fields, asdict

@dataclass
class ModelConfig:
    """
    Data container for model configurations.
    Uses Python dataclass to automatically generate __init__, __repr__, and __eq__.
    """
    modelType: str = ''
    temperature: float = 0
    modelName: str = ''
    displayName: str = ''

    @classmethod
    def from_args(cls, args):
        """
        Factory method to create a ModelConfig instance from argparse results.
        
        It filters the arguments to ensure:
        1. Only fields defined in the dataclass are included (to avoid TypeError).
        2. None values are skipped (to preserve the dataclass default values).
        """
        # Get a set of all valid field names defined in this dataclass
        valid_keys = {f.name for f in fields(cls)}
        
        # Construct a dictionary by filtering the vars(args) dictionary
        # vars(args) converts the Namespace object to a dictionary
        filtered_data = {
            key: value 
            for key, value in vars(args).items() 
            if key in valid_keys and value is not None
        }

        # Unpack the filtered dictionary into the class constructor
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
        """
        return asdict(self)