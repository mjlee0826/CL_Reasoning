from Model.ModelConfig import ModelConfig
from Model.ModelType import MODEL_TO_DISPLAYNAME, MODEL_TO_DEFAULT_MODELNAME
from Model.LLMResponse import LLMResponse

import time
import openai

# Transient API failures that generate() retries with exponential backoff
RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)

class Model():
    """
    Base class for all LLM implementations.
    Acts as a wrapper that delegates configuration access to a ModelConfig object.
    """
    # Whether _complete forwards `seed` to the provider (recorded in arm file metadata)
    SUPPORTS_SEED = True

    def __init__(self, config: ModelConfig):
        """
        Initialize the model with a specific configuration container.
        
        Args:
            config (ModelConfig): An instance containing model parameters (e.g., temperature, modelName).
        """
        self.config: ModelConfig = config
        self.config.displayName = MODEL_TO_DISPLAYNAME[self.config.modelType].value
        self.config.modelName = self.config.modelName or MODEL_TO_DEFAULT_MODELNAME[self.config.modelType].value
    
    def __getattr__(self, name):
        """
        Dynamic delegation logic.
        If an attribute is not found in the Model instance, 
        it searches for it in the self.config object.
        
        Example: Accessing model.temperature will return self.config.temperature.
        """
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

    def generate(self, messages: list, temperature: float = None, seed: int = None, max_retries: int = 6) -> LLMResponse:
        """
        Template method used by the Generate / Aggregate strategies.
        Handles retries and response unpacking; concrete models only implement _complete().

        Args:
            messages (list): Chat messages, e.g. [{"role": "user", "content": prompt}].
            temperature (float): Overrides ModelConfig.temperature when not None (sampling belongs to the arm).
            seed (int): Passed to the provider when not None.

        Returns:
            LLMResponse: error is set instead of raising when the call keeps failing.
        """
        temperature = self.temperature if temperature is None else temperature

        for attempt in range(max_retries):
            try:
                response = self._complete(messages, temperature, seed)
                usage = getattr(response, "usage", None)
                return LLMResponse(
                    text=response.choices[0].message.content or "",
                    model_version=getattr(response, "model", "") or "",
                    usage_in=getattr(usage, "prompt_tokens", None),
                    usage_out=getattr(usage, "completion_tokens", None),
                )
            except RETRYABLE_ERRORS as e:
                if attempt == max_retries - 1:
                    return LLMResponse(error=f"{type(e).__name__}: {e}")
                # 指數退避：5, 10, 20, 40, 80 秒，上限 180 秒
                wait_time = min(5 * 2 ** attempt, 180)
                print(f"[{self.displayName}] {type(e).__name__}，等待 {wait_time} 秒後進行第 {attempt + 1} 次重試...")
                time.sleep(wait_time)
            except Exception as e:
                # Non-transient errors (bad request, auth, malformed response) are not retried
                return LLMResponse(error=f"{type(e).__name__}: {e}")

        return LLMResponse(error="max_retries must be >= 1")

    def _complete(self, messages: list, temperature: float, seed: int):
        """
        Sends one chat completion request and returns the raw provider response
        (an OpenAI-compatible ChatCompletion). Must NOT swallow exceptions.
        Overridden by concrete subclasses.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support generate()")

    def countTokens(self, text: str) -> int:
        """
        getTokenLens with a per-instance memo. Debate re-sends the same history every round and
        Gemini counts tokens through an API call, so identical texts are only counted once.
        """
        if not text:
            return 0
        # __dict__ is used directly because __getattr__ delegates unknown attributes to the config
        cache = self.__dict__.setdefault("_token_cache", {})
        if text not in cache:
            cache[text] = self.getTokenLens(text)
        return cache[text]