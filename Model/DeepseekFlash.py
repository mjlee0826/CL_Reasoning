from openai import OpenAI
from transformers import AutoTokenizer
from Model.Model import Model
from Model.ModelConfig import ModelConfig
import os


class DeepseekFlash(Model):
    """
    DeepSeek's official API, model "deepseek-flash" = DeepSeek-V4.1-Flash (API docs, 2026-10-04; the official API no
    longer serves V4-Flash). The legacy V3.2 data comes from Model/Deepseek.py instead.

    - Thinking is on by default for this model; it is disabled, so the arms are non-thinking like GPT-4o mini and
      Qwen3-8B (enable_thinking False).
    - seed is accepted.
    - The response only names the alias ("deepseek-flash"), so model_version_string appends the version the docs map
      it to. DeepSeek has moved its aliases to newer models before, so run all arms of an experiment together.
    """
    BASE_URL = "https://api.deepseek.com"
    SERVED_VERSION = "DeepSeek-V4.1-Flash"
    TOKENIZER = "deepseek-ai/DeepSeek-V4.1-Flash"
    MAX_TOKENS = 8192

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("[DeepseekFlash] DEEPSEEK_API_KEY is not set: token counting works, API calls will fail")
        # An explicit placeholder keeps the OpenAI SDK from falling back to OPENAI_API_KEY and sending it to DeepSeek
        self.client = OpenAI(api_key=api_key or "DEEPSEEK_API_KEY-not-set", base_url=self.BASE_URL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER, trust_remote_code=True)

    def _complete(self, messages, temperature, seed):
        kwargs = dict(model=self.modelName, messages=messages, max_tokens=self.MAX_TOKENS, temperature=temperature,
                      stream=False, extra_body={"thinking": {"type": "disabled"}})
        if seed is not None:
            kwargs["seed"] = seed
        return self.client.chat.completions.create(**kwargs)

    def _versionString(self, response) -> str:
        return f"{super()._versionString(response)}@{self.SERVED_VERSION}"

    def getTokenLens(self, text: str):
        return len(self.tokenizer.encode(text))
