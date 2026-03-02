from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetType import DATASET_TO_DISPLAYNAME

class Dataset():
    """
    Base class for all Dataset implementations.
    This class acts as a wrapper around DatasetConfig and provides common utilities 
    for data retrieval and answer comparison.
    """
    def __init__(self, config: DatasetConfig):
        """
        Initialize the Dataset with a specific configuration.
        
        Args:
            config (DatasetConfig): The configuration object containing dataset parameters.
        """
        self.config = config
        
        # Initialize display name from the mapping dictionary.
        # Accessing .value because DATASET_TO_DISPLAYNAME maps to Enum objects.
        self.config.displayName = DATASET_TO_DISPLAYNAME[self.config.datasetType].value

        # Placeholder for the actual data records (usually a list of dictionaries).
        self.data: list[dict] = []
        
    @staticmethod
    def compareTwoAnswer(answer1, answer2) -> bool:
        """
        Static utility to compare two answers for equality.
        
        Returns:
            bool: True if answers match, False otherwise.
        """
        return answer1 == answer2
    
    def __getattr__(self, name):
        """
        Dynamic delegation logic.
        If an attribute (like 'nums' or 'sample') is not found in this Dataset instance, 
        it will be automatically retrieved from the self.config object.
        
        Example: 
            Calling dataset.nums is equivalent to dataset.config.nums.
        """
        return getattr(self.config, name)

    def getData(self) -> list:
        """
        Retrieves a subset of the data based on the configured 'nums' 
        and multiplies it by the 'sample' count.
        
        Returns:
            list: A repeated list of data records for evaluation.
        """
        # Note: If self.nums is -1, Python slicing [0:-1] will exclude the last item.
        # Consider adding a check if self.nums == -1 to return the full list.
        return self.data[0:self.nums] * self.sample
    
    def getDataById(self, id: int):
        """
        Retrieves a specific data record by its index.
        
        Args:
            id (int): The index of the record to fetch.
            
        Returns:
            dict or None: The data record if the ID is within bounds, else None.
        """
        # Note: Added parentheses to self.getNums() as it is a method in DatasetConfig
        # or accessed via delegation if it's a property.
        if id >= self.nums:
            return None
        return self.data[id]