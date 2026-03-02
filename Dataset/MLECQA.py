from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from datasets import load_dataset

class MLECQA(Dataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        dataset = load_dataset("shuyuej/MLEC-QA-Benchmark", split="test")
        
        for idx, data in enumerate(dataset):
            question = self.createQuestion(data["qtext"], data["options"])
            ans = data["answer"]

            self.data.append({
                "id": idx,
                "question": question,
                "answer": ans
            })
        
        if self.nums == -1 or self.nums > len(self.data):
            self.config.nums = len(self.data)


    def createQuestion(self, question, choices):
        choicesPrompt = ""
        
        for key, value in choices.items():
            choicesPrompt += f'{key}: {value}\n'

        prompt = f'There is a Problem: \n{question}.\n' \
        f'And there are {len(choices)} choices\n' \
        f'{choicesPrompt}' \
        f'Please choose a choice based on the question' \
        f'At the end of your response, provide your answer in this exact JSON format: \n' \
        f'{{"answer": "your_letter_choice"}}\n' \
        f'The answer must be a single choice and only one English letter (A-Z). You cannot output other letters. You have to output double quotation marks. You have to ouput only one line.\n'
        return prompt