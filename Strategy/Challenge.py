from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from File.File import File  # 引入你封裝好的 File 類別

from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Strategy.PromptAbstractFactory.PromptTwoResultCOTFactory import PromptTwoResultCOTFactory
from Strategy.PromptAbstractFactory.PromptDebateCOTFactory import PromptDebateCOTFactory
from tqdm import tqdm

class Challenge(Strategy):
    """
    Implements a 'Challenge' (or Debate) strategy where two agents (representing two different 
    languages or prompting techniques) debate their answers until they reach a consensus.
    If consensus is not reached within a given threshold, a third 'judge' step is invoked.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, file1: File, file2: File, threshold: int = 3):
        super().__init__(config)
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.threshold = threshold

        # Directly store the pre-parsed File objects
        self.file1 = file1
        self.file2 = file2

        # Safely extract languages from the File objects (fallback to english if missing)
        langs1 = self.file1.getLanguage()
        langs2 = self.file2.getLanguage()
        self.lang1 = langs1[0] if isinstance(langs1, list) and langs1 else "english"
        self.lang2 = langs2[0] if isinstance(langs2, list) and langs2 else "english"

        # Dynamically update the display name to reflect the battling languages
        self.config.displayName = f"Challenge ({self.lang1} vs {self.lang2})"

    def getDebatePrompt(self, target_lang: str, opponent_answer: str) -> str:
        """Constructs the prompt used during the back-and-forth debate phase."""
        prompt = PromptDebateCOTFactory().getPrompt(target_lang, opponent_answer) + PromptFormatFactory().getPrompt(target_lang)
        return prompt
    
    def getJudgePrompt(self, target_lang: str, question: str, result1: str, result2: str) -> str:
        """Constructs the prompt for the final judge if consensus is not reached."""
        prompt = PromptTwoResultCOTFactory().getPrompt(target_lang, question, result1, result2, self.lang1, self.lang2) \
                + PromptFormatFactory().getPrompt(target_lang)
        return prompt
    
    def runChallenge(self, question1: str, question2: str, result1: str, result2: str, answer1: str, answer2: str):
        """
        Executes the debate loop between the two agents.
        """
        # Initialize chat histories
        record1 = [
            {"role": "user", "content": question1},
            {"role": "assistant", "content": result1}
        ]
        record2 = [
            {"role": "user", "content": question2},
            {"role": "assistant", "content": result2}
        ]
        
        answerRecord1, answerRecord2 = [answer1], [answer2]
        cur_turn = 0

        # Loop until answers match or threshold is reached
        while not self.dataset.compareTwoAnswer(answer1, answer2) and cur_turn < self.threshold:
            # Generate debate prompts using the opponent's previous full result
            prompt1 = self.getDebatePrompt(self.lang1, result2)
            prompt2 = self.getDebatePrompt(self.lang2, result1)

            record1.append({"role": "user", "content": prompt1})
            record2.append({"role": "user", "content": prompt2})

            # Get new responses using conversational history
            result1 = self.model.getListRes(record1)
            result2 = self.model.getListRes(record2)

            record1.append({"role": "assistant", "content": result1})
            record2.append({"role": "assistant", "content": result2})

            # Parse the new extracted answers
            answer1, answer2 = self.parseAnswer(result1), self.parseAnswer(result2)
            answerRecord1.append(answer1)
            answerRecord2.append(answer2)

            cur_turn += 1
            
        return record1, record2, result1, result2, answer1, answer2, answerRecord1, answerRecord2, cur_turn
    
    def getRes(self) -> list:
        """
        Executes the main evaluation loop. Pre-calculates discrepancies to show 
        an accurate progress bar, then iterates through the dataset to perform debates.
        """
        self.log.logInfo(self, self.model, self.dataset, self.file1, self.file2)

        result = [{
            "Model": self.model.config.to_dict(),
            "Dataset": self.dataset.config.to_dict(),
            "Strategy": self.config.to_dict(),
            "File1": {
                "path": self.file1.file_path,
                "Model": self.file1.getModelConfig().to_dict(),
                "Dataset": self.file1.getDatasetConfig().to_dict(),
                "Strategy": self.file1.getStrategyConfig().to_dict(),
            },
            "File2": {
                "path": self.file2.file_path,
                "Model": self.file2.getModelConfig().to_dict(),
                "Dataset": self.file2.getDatasetConfig().to_dict(),
                "Strategy": self.file2.getStrategyConfig().to_dict(),
            }
        }]

        database = self.dataset.getData()
        discrepancy_ids = []
        
        # 1. Pre-calculate discrepancies using the File class's O(1) ID lookup
        for data in database:
            q_id = data["id"]
            rec1 = self.file1.getRecordById(q_id)
            rec2 = self.file2.getRecordById(q_id)
            
            # Skip if the record is missing in either file
            if not rec1 or not rec2:
                continue
                
            ans1 = rec1.get("MyAnswer", "")
            ans2 = rec2.get("MyAnswer", "")
            
            if not self.dataset.compareTwoAnswer(ans1, ans2):
                discrepancy_ids.append(q_id)

        self.log.logMessage(f'Total Discrepancies to Debate: {len(discrepancy_ids)} / {len(database)}')
        pbar = tqdm(total=len(discrepancy_ids), desc="Debating")

        # 2. Main Evaluation Loop
        for data in database:
            q_id = data.get("id")
            
            # Instantly fetch records using the File getter
            rec1 = self.file1.getRecordById(q_id)
            rec2 = self.file2.getRecordById(q_id)
            
            if not rec1 or not rec2:
                continue

            question1, question2 = rec1.get("Question", ""), rec2.get("Question", "")
            result1, result2 = rec1.get("Result", ""), rec2.get("Result", "")
            answer1, answer2 = rec1.get("MyAnswer", ""), rec2.get("MyAnswer", "")
            correct_answer = data.get("answer", "")

            cur_turn = 0
            resultOutput3 = ""
            myAnswer = ""
            record1, record2 = [], []
            answerRecord1, answerRecord2 = [], []

            # Case A: They already agree. No debate needed.
            if self.dataset.compareTwoAnswer(answer1, answer2):
                myAnswer = answer1

            # Case B: They disagree. Initiate Challenge/Debate.
            else:
                record1, record2, result1, result2, answer1, answer2, answerRecord1, answerRecord2, cur_turn = \
                    self.runChallenge(question1, question2, result1, result2, answer1, answer2) 

                self.log.logMessage(f'Record1：\n{record1}')
                self.log.logMessage(f'結果1：\n{answerRecord1}')
                self.log.logMessage(f'Record2：\n{record2}')
                self.log.logMessage(f'結果2：\n{answerRecord2}')
                self.log.logMessage(f'Times：\n{cur_turn}')
                
                # Check if consensus was reached after the debate
                if self.dataset.compareTwoAnswer(answer1, answer2):
                    myAnswer = answer1
                    self.log.logMessage(f'Result: Agents reached consensus!')
                else:
                    # Case C: Still disagree after threshold. Call the Judge.
                    judge_lang = getattr(self.dataset.config, "language", "english")
                    resultOutput3 = self.model.getRes(self.getJudgePrompt(judge_lang, question1, result1, result2))
                    myAnswer = self.parseAnswer(resultOutput3)
                    self.log.logMessage(f'Result 3 (Judge): \n{resultOutput3}')

                self.log.logMessage(f'My Answer: {myAnswer} | Correct Answer: {correct_answer}\n')
                pbar.update()

            # Append the comprehensive debate log to the final result
            result.append({
                "id": q_id,
                "Question1": question1,
                "Question2": question2,
                "Record1": record1,
                "Record2": record2,
                "AnswerRecord1": answerRecord1,
                "AnswerRecord2": answerRecord2,
                "Times": cur_turn,
                "Result3": resultOutput3,
                "Answer": correct_answer,
                "MyAnswer": myAnswer
            })
        
        pbar.close()
        return result
    
    @staticmethod
    def getTokenLens(model: Model, data):
        """Calculates total token usage across all debate turns and the judge phase."""
        result = model.getTokenLens(data.get("Question1", "")) + model.getTokenLens(data.get("Question2", ""))
        
        for r in data.get("Record1", []):
            if r.get("role") == "assistant":
                # Assuming symmetric token usage tracking, calculate token cost for content
                result += model.getTokenLens(r.get("content", "")) * 2 
                
        if data.get("Result3"):
            result += model.getTokenLens(data["Result3"])
            
        return result