import google.generativeai as genai
from Model.Model import Model
from Model.ModelType import ModelType, MODEL_TO_NAME
import os

class Gemma(Model):
    NAME = MODEL_TO_NAME[ModelType.GEMMA]

    def __init__(self, temperature, modelName):
        if modelName == None:
            modelName = "models/gemma-3-27b-it"
            print(f"Log: 'modelName' default to {modelName}.")

        super().__init__(temperature, modelName)
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
