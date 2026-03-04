from dataclasses import dataclass, fields, asdict, field

@dataclass
class StrategyConfig:
    """
    Data container for strategy configurations.
    """
    strategyType: str = ''
    displayName: str = ''
    
    # which would cause all instances to share the same list in memory.
    languages: list[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args):
        """
        Factory method to instantiate config from command-line arguments.
        """
        valid_keys = {f.name for f in fields(cls)}
        
        filtered_data = {
            key: value 
            for key, value in vars(args).items() 
            if key in valid_keys and value is not None
        }

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
    
    def to_dict(self) -> dict:
        return asdict(self)