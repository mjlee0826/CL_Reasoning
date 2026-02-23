from openai import OpenAI
from Model.Model import Model
from Model.ModelType import ModelType, MODEL_TO_NAME
from transformers import AutoTokenizer
import os

class QWEN(Model):
    NAME = MODEL_TO_NAME[ModelType.QWEN]
    
    def __init__(self, tempature, modelName):
        if modelName == None:
            modelName = "qwen3-8b"
            print(f"Log: 'modelName' default to {modelName}.")

        super().__init__(tempature, modelName)
        self.name: str = QWEN.NAME
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            trust_remote_code=True
        )
    
    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                extra_body={"enable_thinking": False},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.tempature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in QWEN model: {e}"

    def getListRes(self, promptList):
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                extra_body={"enable_thinking": False},
                messages=promptList,
                max_tokens=8192,
                temperature=self.tempature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in QWEN model: {e}"
        
    def getTokenLens(self, text: str):
        return len(self.tokenizer.encode(text))