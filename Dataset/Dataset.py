from Dataset.DatasetConfig import DatasetConfig
from Dataset.DatasetType import DATASET_TO_DISPLAYNAME
from Dataset.path import translatedBaseDir

import json
import os

class Dataset():
    """
    Base class for all Dataset implementations.
    This class acts as a wrapper around DatasetConfig and provides common utilities 
    for data retrieval, translation application, and answer comparison.
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

        # Define the default original language of the dataset.
        # This is used as a baseline to check whether a translation file needs to be loaded.
        self.oriDataLanguage = 'english'
        
    @staticmethod
    def compareTwoAnswer(answer1, answer2) -> bool:
        """
        Static utility to compare two answers for equality.
        Can be overridden by subclasses for more complex evaluation metrics (e.g., comparing math formulas).
        
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
    
    def _apply_translation(self):
        """
        Shared logic for applying multi-language translations.
        Must be called by subclasses at the end of their __init__ method, 
        AFTER the original data has been fully loaded into self.data.
        
        This method uses a "double loading" mechanism: it loads the translated JSON file 
        and replaces the original questions while preserving the original ground truth answers.
        """
        # Get the target language from config. Fallback to the original language (e.g., english)
        target_lang = getattr(self.config, 'language', self.oriDataLanguage).lower()
        
        # Skip translation process if the target language matches the original language or is empty
        if target_lang == self.oriDataLanguage or not target_lang:
            return

        # Format the expected translation file name based on dataset type and target language
        lang_capitalized = target_lang.capitalize()
        translated_file = f"{self.config.datasetType}_{lang_capitalized}.json"
        translated_path = os.path.join(translatedBaseDir, translated_file)

        try:
            with open(translated_path, "r", encoding="utf-8") as f:
                translated_data = json.load(f)

            # Create a hash map mapping 'id' to the 'Translated' string for O(1) fast lookup.
            # We skip the first element [0] because it contains the configuration metadata dict.
            trans_map = {item["id"]: item.get("Translated", "") for item in translated_data[1:]}

            # Overwrite the original questions in self.data with the translated ones
            for item in self.data:
                if item["id"] in trans_map and trans_map[item["id"]]:
                    item["question"] = trans_map[item["id"]]
                    
            print(f"[{self.config.displayName}] Successfully loaded and applied {lang_capitalized} translation.")
            
        except FileNotFoundError:
            # Graceful degradation: fallback to the original language if the translation file is missing
            print(f"[{self.config.displayName}] Warning: Translation file not found at {translated_path}. Using original language.")
        except Exception as e:
            # Catch other potential errors (e.g., JSONDecodeError) to prevent application crash
            print(f"[{self.config.displayName}] Error loading translation: {e}")

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
        # Ensure the requested id does not exceed the allowed subset limit
        if id >= self.nums:
            return None
        return self.data[id]