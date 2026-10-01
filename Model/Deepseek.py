from Model.Model import Model
from openai import OpenAI
from transformers import AutoTokenizer
from Model.ModelConfig import ModelConfig
import os

# The legacy results were generated through DeepSeek's official API as "deepseek-chat", which pointed to
# DeepSeek-V3.2 at the time (2026-03). The official API has since retired V3.2 ("deepseek-chat" now serves
# the V4 family), so V3.2 (open weights, MIT) is served by GMI Cloud's OpenAI-compatible API instead.
GMI_BASE_URL = "https://api.gmi-serving.com/v1"

# Configs rebuilt from legacy result files still carry the old official alias
LEGACY_MODEL_NAMES = {"deepseek-chat": "deepseek-ai/DeepSeek-V3.2"}

class Deepseek(Model):
    # Appended to response.model in model_version_string: the model id alone does not say who served it
    PROVIDER = "GMI"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config.modelName = LEGACY_MODEL_NAMES.get(self.config.modelName, self.config.modelName)

        api_key = os.getenv('GMI_API_KEY')
        if not api_key:
            print("[Deepseek] GMI_API_KEY is not set: token counting works, API calls will fail")

        # An explicit placeholder keeps the OpenAI SDK from falling back to OPENAI_API_KEY and sending it to GMI
        self.client = OpenAI(
            api_key=api_key or "GMI_API_KEY-not-set",
            base_url=GMI_BASE_URL
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3",
            trust_remote_code=True
        )

    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in Deepseek model: {e}"

    def getListRes(self, promptList):
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=promptList,
                max_tokens=8192,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in DeepSeek model: {e}"

    def _complete(self, messages, temperature, seed):
        kwargs = dict(model=self.modelName, messages=messages, max_tokens=8192, temperature=temperature, stream=False)
        if seed is not None:
            kwargs["seed"] = seed
        return self.client.chat.completions.create(**kwargs)

    def _versionString(self, response) -> str:
        return f"{super()._versionString(response)}@{self.PROVIDER}"

    def getTokenLens(self, text: str):
        return len(self.tokenizer.encode(text))
