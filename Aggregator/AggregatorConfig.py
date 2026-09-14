from dataclasses import dataclass, fields, asdict

@dataclass
class AggregatorConfig:
    """
    Data container for aggregator configurations.
        k          number of candidate arms the aggregator expects
        seed       vote tie-breaks and the judge's balanced presentation order
        threshold  maximum debate rounds before the judge is called (Debate only)
    """
    aggregatorType: str = ''
    displayName: str = ''
    k: int = 2
    seed: int = 0
    threshold: int = 3

    @classmethod
    def from_dict(cls, data_dict: dict):
        """
        從 JSON 字典建立 Config 實例。
        會自動過濾掉 data_dict 中不屬於此 dataclass 的 key，避免 TypeError。
        """
        valid_keys = {f.name for f in fields(cls) if f.init}
        return cls(**{key: value for key, value in data_dict.items() if key in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)
