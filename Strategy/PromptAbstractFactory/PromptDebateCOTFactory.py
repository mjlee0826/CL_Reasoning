from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory

class PromptDebateCOTFactory(PromptAbstractFactory):
    def __init__(self):
        super().__init__()

    def englishPrompt(self, new_answer):
        prompt = f'Could the answer you just provided be incorrect? Is it possible that the correct answer is actually the following:\n' \
            f'```\n{new_answer}\n```\n' \
            f'Analyze the reasoning process of both answers. Think step by step and output your thought process.\n'
        return prompt    

    def chinesePrompt(self, new_answer):
        prompt = f'你剛剛提供的答案有沒有可能是錯誤的? 正確答案有沒有可能是以下這份答案?\n' \
            f'```\n{new_answer}\n```\n' \
            f'分析兩個答案的推理過程，一步一步思考並輸出思考過程\n'
        return prompt
    
    def spanishPrompt(self, new_answer):
        prompt = f'¿Podría la respuesta que acaba de proporcionar ser incorrecta? ¿Es posible que la respuesta correcta sea en realidad la siguiente:\n' \
            f'```\n{new_answer}\n```\n' \
            f'Analice el proceso de razonamiento de ambas respuestas. Piense paso a paso y exponga su proceso de pensamiento.\n'
        return prompt

    def japanesePrompt(self, new_answer):
        prompt = f'先ほど提供した回答が間違っている可能性はありますか？正しい回答が実際には次のものである可能性はありますか：\n' \
            f'```\n{new_answer}\n```\n' \
            f'両方の回答の推論プロセスを分析してください。段階的に考え、思考プロセスを出力してください。\n'
        return prompt

    def russianPrompt(self, new_answer):
        prompt = f'Может ли предоставленный вами ответ быть неверным? Возможно ли, что правильный ответ на самом деле следующий:\n' \
            f'```\n{new_answer}\n```\n' \
            f'Проанализируйте процесс рассуждения обоих ответов. Думайте шаг за шагом и выведите свой процесс мышления.\n'
        return prompt