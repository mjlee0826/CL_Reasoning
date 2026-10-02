from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from File.ResultStore import ResultStore
from Strategy.PromptAbstractFactory.PromptRewriteFactory import PromptRewriteFactory
from Strategy.PromptAbstractFactory.PromptRewriteAgainFactory import PromptRewriteAgainFactory

from tqdm import tqdm
import re

# Answer-option lines produced by Dataset.createQuestion: "A: ..." (MMLU, TruthfulQA),
# "A) ..., B) ..." (CommonsenseQA) and "a ) ..., b ) ..." (MathQA)
OPTION_LINE = re.compile(r"^\s*([A-Z]: |[A-Z]\) |a \) )")

# A markdown code fence around the whole output (the prompt shows the question inside ``` fences and the
# model sometimes mirrors them)
OUTER_FENCE = re.compile(r"^```[a-zA-Z]*[ \t]*\n(.*)\n[ \t]*```$", re.S)


class Rewrite(Strategy):
    """
    Paraphrases every English question with a single fixed "rewriter" model (run_rewrite.py).

    Version 1 rewords the stem while keeping numbers / names / units / answer options verbatim.
    Version n >= 2 also sees versions 1..n-1 (`previous`) and must use different wording from all of
    them. The output file ([meta, {id, Question, Rewritten}, ...]) is read by Dataset._apply_rewrite().

    Records go through a ResultStore: a failed API call is not written, so rerunning the same command
    fills it in, and an error message can never end up as a "rewritten question".

    A fence wrapping the whole output is removed first (stripOuterFence), so every rewrite reaches the
    generation prompt in the same plain format.
    A rewrite must keep every answer option and the answer-format placeholder (preservesOptions).
    If it does not, the model gets one follow-up turn to output the full question again; a rewrite
    that still drops them is not written and is reported as failed.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log,
                 store: ResultStore = None, previous: dict = None):
        super().__init__(config)

        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.store: ResultStore = store
        # {id: [rewrite version 1, ..., version n-1]}; empty for version 1
        self.previous: dict = previous or {}

        # Prevents IndexError if config.languages is not provided.
        if self.config.languages:
            self.config.displayName += f" ({self.config.languages[0]})"
        else:
            self.config.displayName += " (english)"

    def getPrompt(self, question: str, previous: list[str] = None) -> str:
        """Constructs the rewrite prompt using the Factory pattern."""
        target_lang = self.config.languages[0] if self.config.languages else "english"
        if previous:
            return PromptRewriteAgainFactory().getPrompt(target_lang, question, previous)
        return PromptRewriteFactory().getPrompt(target_lang, question)

    def getRes(self) -> list:
        """
        Rewrites every item missing from the store. Returns the ids whose API call failed (still missing).
        """
        self.log.logInfo(self, self.model, self.dataset)

        if not self.store.metadata:
            self.store.metadata = {
                "Model": self.model.config.to_dict(),
                "Dataset": self.dataset.config.to_dict(),
                "Strategy": self.config.to_dict(),
            }

        validation = self.store.metadata.setdefault("validation", {"repair_turns": 0, "rejected": 0})
        validation.setdefault("fence_stripped", 0)
        todo = [data for data in self.dataset.getData() if not self.store.has(data["id"])]
        failed = []
        for data in tqdm(todo, desc=f"Rewrite v{self.config.rewriteVersion}"):
            question = data["question"]
            messages = [{"role": "user", "content": self.getPrompt(question, self.previous.get(data["id"]))}]
            response = self.model.generate(messages)
            self.store.addUsage(response)
            text = self.stripOuterFence(response.text)

            if response.ok and text.strip() and not self.preservesOptions(question, text):
                # T=0 would repeat the same output on a rerun, so ask once more within the conversation
                messages += [{"role": "assistant", "content": response.text},
                             {"role": "user", "content": PromptRewriteAgainFactory.REPAIR_MESSAGE}]
                response = self.model.generate(messages)
                self.store.addUsage(response)
                text = self.stripOuterFence(response.text)
                validation["repair_turns"] += 1

            if not response.ok or not text.strip():
                failed.append(data["id"])
                self.log.logMessage(f'API error on item {data["id"]}: {response.error}')
                continue

            if not self.preservesOptions(question, text):
                failed.append(data["id"])
                validation["rejected"] += 1
                self.log.logMessage(f'Item {data["id"]}: rewrite dropped answer options or instructions; not written')
                continue

            validation["fence_stripped"] += text != response.text
            self.store.add({
                "id": data["id"],
                "Question": data["question"],
                "Rewritten": text,
            })
            self.log.logMessage(f'改寫問題 (Rewritten)：\n{text}')

        self.store.save()
        return failed

    @staticmethod
    def stripOuterFence(text: str) -> str:
        """Removes a code fence wrapping the whole text; a fence anywhere else is left untouched."""
        match = OUTER_FENCE.match(text.strip())
        if match and "```" not in match.group(1):
            return match.group(1)
        return text

    @staticmethod
    def preservesOptions(question: str, rewritten: str) -> bool:
        """
        True when every answer-option line of the original question, and the answer-format placeholder
        if the question has one, appear verbatim in the rewrite (whitespace-normalized).
        """
        norm = lambda s: " ".join(str(s).split())
        text = norm(rewritten)
        options = [norm(line) for line in question.split("\n") if OPTION_LINE.match(line)]
        if any(option not in text for option in options):
            return False
        return "your_letter_choice" not in question or "your_letter_choice" in rewritten

    @staticmethod
    def getTokenLens(model: Model, data):
        """Calculate token usage for the rewritten text."""
        return model.getTokenLens(data.get("Rewritten", ""))
