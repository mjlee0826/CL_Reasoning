from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory

class PromptTwoResultCOTFactory(PromptAbstractFactory):
    def __init__(self):
        super().__init__()

    def englishPrompt(self, question, answer1, answer2, lang1, lang2):
        return f'For the following question\n```\n{question}\n```\nThere is a {lang1} answer as follows\n' \
            f'```\n{answer1}\n```\n' \
            f'And an {lang2} answer as follows\n' \
            f'```\n{answer2}\n```\n' \
            f'Based on the question, select and output a more correct answer.' \
            f'You must think step by step about which parts of the reasoning in the {lang1} answer and {lang2} answer are incorrect, and output your reasoning process.\n'
    
    def chinesePrompt(self, question, answer1, answer2, lang1, lang2):
        return f'針對以下問題\n```\n{question}\n```\n有一個 {lang1} 答案如下\n' \
            f'```\n{answer1}\n```\n' \
            f'以及一個 {lang2} 答案如下\n' \
            f'```\n{answer2}\n```\n' \
            f'請根據問題選出並輸出較正確的答案。' \
            f'你必須一步步分析 {lang1} 答案和 {lang2} 答案中推理錯誤的部分，並輸出你的推理過程。\n'
    
    def spanishPrompt(self, question, answer1, answer2, lang1, lang2):
        return f'Para la siguiente pregunta\n```\n{question}\n```\nHay una respuesta {lang1} como sigue\n' \
            f'```\n{answer1}\n```\n' \
            f'Y una respuesta {lang2} como sigue\n' \
            f'```\n{answer2}\n```\n' \
            f'Basado en la pregunta, selecciona y muestra la respuesta más correcta.' \
            f'Debes pensar paso a paso sobre qué partes del razonamiento en la respuesta {lang1} y la respuesta {lang2} son incorrectas, y mostrar tu proceso de razonamiento.\n'

    def japanesePrompt(self, question, answer1, answer2, lang1, lang2):
        return f'以下の質問に対して\n```\n{question}\n```\n次のような{lang1}の回答があります\n' \
            f'```\n{answer1}\n```\n' \
            f'そして次のような{lang2}の回答があります\n' \
            f'```\n{answer2}\n```\n' \
            f'質問に基づき、より正しい回答を選択して出力してください。' \
            f'{lang1}の回答と{lang2}の回答の推論のどの部分が間違っているかを段階的に考え、あなたの推論プロセスを出力する必要があります。\n'

    def russianPrompt(self, question, answer1, answer2, lang1, lang2):
        return f'Для следующего вопроса\n```\n{question}\n```\nЕсть ответ на {lang1} следующим образом\n' \
            f'```\n{answer1}\n```\n' \
            f'И ответ на {lang2} следующим образом\n' \
            f'```\n{answer2}\n```\n' \
            f'На основе вопроса выберите и выведите более правильный ответ.' \
            f'Вы должны пошагово подумать о том, какие части рассуждений в ответе на {lang1} и ответе на {lang2} являются неверными, и вывести процесс ваших рассуждений.\n'