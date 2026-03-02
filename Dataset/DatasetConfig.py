from dataclasses import dataclass, fields, asdict, field

@dataclass
class DatasetConfig:
    """
    Data container for dataset configurations.
    """
    datasetType: str = ''
    displayName: str = ''
    nums: int = -1
    sample: int = 1
    
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
    
    def to_dict(self) -> dict:
        """
        Convert the dataclass instance to a standard dictionary.
        This will include the calculated 'dataNums' field as well.
        """
        return asdict(self)