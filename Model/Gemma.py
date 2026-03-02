import google.generativeai as genai
from Model.Model import Model
from Model.ModelConfig import ModelConfig
import os

class Gemma(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name: str = Gemma.NAME
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel(self.modelName)
    
    def getRes(self, prompt) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=4096
                )
            )
            return response.text
        except Exception as e:
            return f"Error in Gemma model: {e}"
