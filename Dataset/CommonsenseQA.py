from Dataset.Dataset import Dataset
from Dataset.path import commensenseqa_path
from Dataset.DatasetType import DATASET_TO_NAME, DatasetType
import json

class CommonsenseQA(Dataset):
    NAME = DATASET_TO_NAME[DatasetType.COMMONSENSEQA]
    
    def __init__(self, nums=-1, sample=1):
        super().__init__(nums, sample)
        self.name: str = CommonsenseQA.NAME

        with open(commensenseqa_path, "r") as f:
            originData = json.load(f)

        for idx, odata in enumerate(originData):
            question = self.createQuestion(odata["question"], odata["choices"]["label"], odata["choices"]["text"])
            ans = odata["answerKey"]
            self.data.append({
                "id": idx,
                "question": question,
                "answer": ans
            })

        if self.nums == -1 or self.nums > len(self.data):
            self.nums = len(self.data)

    def createQuestion(self, question, labels, texts) -> str:
        # format choices as: A) ignore, B) enforce, ...
        formatted_choices = ", ".join(
            [f"{label}) {text}" for label, text in zip(labels, texts)]
        )
        result = (
            f"There is a Question: \n{question}\n"
            f"And there are {len(labels)} choices:\n"
            f"{formatted_choices}\n"
            f"Please choose a choice based on the question.\n"
            f"At the end of your response, provide your answer in this exact JSON format: \n"
            f'{{"answer": "your_letter_choice"}}\n'
            f"The answer must be a single English letter ({labels[0]}-{labels[-1]}). You have to output double quotation marks. You have to ouput only one line.\n"
        )
        return result
