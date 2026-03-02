from Dataset.Dataset import Dataset
from Dataset.DatasetType import DatasetType, get_dataset_map

class DatasetFactory():
    """
    Factory class responsible for instantiating specific Dataset objects.
    It utilizes a dynamic mapping system to avoid tight coupling with 
    concrete dataset implementations.
    """
    def __init__(self):
        # Dictionary to store the mapping of types to classes
        # Loaded once or on-demand via get_dataset_map()
        pass

    def buildDataset(self, dataset_type: DatasetType, *args, **kwargs) -> Dataset:
        """
        Creates and returns an instance of the specified dataset.
        
        Args:
            dataset_type (DatasetType): The identifier for the dataset to build.
            *args: Variable arguments for the dataset constructor.
            **kwargs: Keyword arguments (usually a DatasetConfig object).
            
        Returns:
            Dataset: An instance of a concrete Dataset subclass, or None if not found.
        """
        # Fetch the mapping (uses lazy imports internally)
        dataset_map = get_dataset_map()
        
        # Retrieve the class based on the Enum key.
        # Since DatasetType inherits from str, this works with both Enum and string values.
        dataset_cls = dataset_map.get(dataset_type)

        if not dataset_cls:
            # Error handling for unsupported dataset types
            print(f"Error: Dataset '{dataset_type}' doesn't exist in the current factory mapping!")
            return None

        # Instantiate and return the specific dataset object
        return dataset_cls(*args, **kwargs)