from Model.ModelConfig import ModelConfig
from Dataset.DatasetConfig import DatasetConfig
from Strategy.StrategyConfig import StrategyConfig

import json
import os

class File:
    """
    Encapsulates the reading and parsing of strategy result JSON files.
    Separates the I/O operations and JSON parsing logic from the Strategy's core business logic.
    """
    def __init__(self, file_path: str):
        """
        Initializes the ResultFile instance and automatically loads the data.
        
        Args:
            file_path (str): The relative or absolute path to the JSON result file.
        """
        self.file_path = file_path
        self.metadata: dict = {}
        self.records_map: dict = {}
        self.languages: list[str] = ["english"]  # Default fallback language

        # Trigger the loading and parsing process upon instantiation
        self._load_and_parse()

    def _load_and_parse(self):
        """
        Reads the JSON file and builds the internal data structures.
        Extracts metadata and constructs an O(1) lookup hash map for the records.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Error: Cannot find result file at {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Ensure the file has at least the metadata block (index 0) and one record (index 1)
        if not raw_data or len(raw_data) < 2:
            raise ValueError(f"Error: File {self.file_path} is empty or missing necessary data.")

        # 1. Parse Metadata (Index 0)
        self.metadata = raw_data[0]
        
        # Extract the language(s) used from the Strategy configuration metadata
        strategy_meta = self.metadata.get("Strategy", {})
        if strategy_meta.get("languages"):
            self.languages = strategy_meta["languages"]

        # 2. Build Hash Map for Records (Index 1 and onwards)
        # Using item.get("id") prevents KeyError if a record is malformed or missing an ID.
        # We only map items that actually have a valid ID.
        self.records_map = {
            item.get("id"): item 
            for item in raw_data[1:] 
            if item.get("id") is not None
        }

    def getRecordById(self, q_id: int) -> dict:
        """
        Retrieves a specific evaluation record by its question ID.
        
        Args:
            q_id (int): The unique identifier of the question.
            
        Returns:
            dict or None: The record dictionary if found, otherwise None.
        """
        return self.records_map.get(q_id)

    def getLanguage(self) -> list[str]:
        """
        Returns the list of languages used to generate this result file.
        
        Returns:
            list[str]: A list of language strings (e.g., ['chinese']).
        """
        return self.languages
    
    def getModelConfig(self) -> ModelConfig:
        """
        Deserializes the 'Model' metadata dictionary into a ModelConfig object.
        Uses .get() to provide an empty dictionary fallback, preventing KeyErrors.
        """
        return ModelConfig.from_dict(self.metadata.get("Model", {}))
    
    def getDatasetConfig(self) -> DatasetConfig:
        """
        Deserializes the 'Dataset' metadata dictionary into a DatasetConfig object.
        """
        return DatasetConfig.from_dict(self.metadata.get("Dataset", {}))
    
    def getStrategyConfig(self) -> StrategyConfig:
        """
        Deserializes the 'Strategy' metadata dictionary into a StrategyConfig object.
        """
        return StrategyConfig.from_dict(self.metadata.get("Strategy", {}))