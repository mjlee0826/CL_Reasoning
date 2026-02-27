from dataclasses import dataclass, fields

@dataclass
class ModelConfig:
    """
    Data container for model configurations.
    Uses Python dataclass to automatically generate __init__, __repr__, and __eq__.
    """
    name: str = ''
    temperature: float = 0
    modelName: str = ''

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