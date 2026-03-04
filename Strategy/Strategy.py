from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.StrategyConfig import StrategyConfig
from Strategy.StrategyType import STRATEGY_TO_DISPLAYNAME
from Log.Log import Log

import re

class Strategy():
    """
    Base class for all testing strategies.
    Provides common parsing utilities and delegates properties to StrategyConfig.
    """
    def __init__(self, config: StrategyConfig):
        self.config: StrategyConfig = config
        
        # This prevents TypeError when concatenating strings later.
        # Also used .get() for safety in case of an invalid strategyType.
        enum_display = STRATEGY_TO_DISPLAYNAME.get(self.config.strategyType)
        if enum_display:
            self.config.displayName = enum_display.value
        else:
            self.config.displayName = "Unknown Strategy"

    def __getattr__(self, name):
        """Dynamic delegation to StrategyConfig."""
        return getattr(self.config, name)
    
    def parseAnswer(self, answer: str) -> str:
        """
        Extracts the core answer from the LLM's raw output.
        Prioritizes JSON format extraction, falling back to Regex.
        """
        result: str = ""
        try:
            # Look for JSON structure: {"answer": "value"}
            json_match = re.findall(r'\{"answer":\s*"([^"]+)"\}', answer)
            if json_match:
                result = json_match[-1]
                return result
            
            # Fallback: Extract the first isolated alphabet character
            match = re.search(r"[A-Za-z](?!.*[A-Za-z])", answer)
            if match:
                result = match.group(1).strip()
        except Exception:
            result = ""
        return result
    
    # The state should be maintained via __init__ injection.
    def getRes(self) -> list:
        """
        Executes the strategy and returns a list of results.
        Must be overridden by subclasses.
        """
        return []
    
    @staticmethod
    def getTokenLens(model: Model, data):
        """Calculates token length for logging and cost estimation."""
        return 0