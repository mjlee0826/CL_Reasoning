from Dataset.Dataset import Dataset
from datasets import load_dataset
from Dataset.DatasetConfig import DatasetConfig
import pandas as pd

class MMLU(Dataset):
    letters = ['A', 'B', 'C', 'D']

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.type: dict = {}

        dataset = load_dataset('cais/mmlu', 'all', split='test')
        for idx, data in enumerate(dataset):
            question = self.createQuestion(data['question'], data['choices'])
            ans = MMLU.letters[data['answer']]

            if not data['subject'] in self.type:
                self.type[data['subject']] = []
            
            data_info = {
                "id": idx,
                "question": question,
                "answer": ans
            }
            self.data.append(data_info)
            self.type[data['subject']].append(data_info)

        if self.nums == -1 or self.nums > len(self.data):
            self.config.nums = len(self.data)
            self.config.dataNums = self.config.nums * self.config.sample
        
        self.realData = self.getRealData()
        self.config.nums = len(self.realData)
        self.config.dataNums = self.config.nums * self.config.sample
        
        self._apply_translation()

    def createQuestion(self, question, choices) -> str:
        choicesPrompt = ""

        for i in range(len(choices)):
            choicesPrompt += f'{MMLU.letters[i]}: {choices[i]}\n'

        result = f'There is a Question: \n{question}\n' \
                f'And there are multiple choices:\n' \
                f'{choicesPrompt}\n' \
                f'Please choose a choice based on the question\n' \
                f'At the end of your response, provide your answer in this exact JSON format: \n' \
                f'{{"answer": "your_letter_choice"}}\n' \
                f'The answer must be a single English letter (A-D). You have to output double quotation marks. You have to ouput only one line.\n'
        return result
    
    def getRealData(self):
        types_list = list(self.type.keys()) 
        base = self.nums // len(types_list)       # 每個至少多少
        remainder =  self.nums % len(types_list)   # 剩下多少要分配

        # 先給每個 base，再把餘數加到前面幾個
        eachNums = [base + 1 if i < remainder else base for i in range(len(types_list))]
        result = []
        for n, t in zip(eachNums, types_list):
            result.extend(self.type[t][0:n])
        return result
    
    def getData(self):
        return self.realData * self.sample