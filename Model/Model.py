from Model.ModelConfig import ModelConfig

class Model():
    """
    Base class for all LLM implementations.
    Acts as a wrapper that delegates configuration access to a ModelConfig object.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model with a specific configuration container.
        
        Args:
            config (ModelConfig): An instance containing model parameters (e.g., temperature, modelName).
        """
        self.config: ModelConfig = config
    
    def __getattr__(self, name):
        """
        Dynamic delegation logic.
        If an attribute is not found in the Model instance, 
        it searches for it in the self.config object.
        
        Example: Accessing model.temperature will return self.config.temperature.
        """
        # Note: Changed 'self.support' to 'self.config' to match your __init__
        return getattr(self.config, name)
    
    def getRes(self, prompt: str) -> str:
        """
        Generate a single response for a given prompt.
        Should be overridden by concrete subclasses (e.g., GPT4omini).
        """
        return ""
    
    def getListRes(self, promptList: list) -> list:
        """
        Process a sequence of conversation messages (chat history) and return the response.
        
        Args:
            promptList (list): A list of dictionaries or strings representing 
            the dialogue history to provide context for the model.
        """
        return []
    
    def getTokenLens(self, text) -> int:
        """
        Calculate the number of tokens in a given text string.
        Implementation varies based on the specific model's tokenizer.
        """
        return 0