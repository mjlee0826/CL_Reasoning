from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Log.Log import Log
from File.ResultStore import ResultStore
from Strategy.PromptAbstractFactory.PromptRewriteFactory import PromptRewriteFactory
from Strategy.PromptAbstractFactory.PromptRewriteAgainFactory import PromptRewriteAgainFactory

from tqdm import tqdm


class Rewrite(Strategy):
    """
    Paraphrases every English question with a single fixed "rewriter" model (run_rewrite.py).

    Version 1 rewords the stem while keeping numbers / names / units / answer options verbatim.
    Version n >= 2 also sees versions 1..n-1 (`previous`) and must use different wording from all of
    them. The output file ([meta, {id, Question, Rewritten}, ...]) is read by Dataset._apply_rewrite().

    Records go through a ResultStore: a failed API call is not written, so rerunning the same command
    fills it in, and an error message can never end up as a "rewritten question".
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

        todo = [data for data in self.dataset.getData() if not self.store.has(data["id"])]
        failed = []
        for data in tqdm(todo, desc=f"Rewrite v{self.config.rewriteVersion}"):
            prompt = self.getPrompt(data["question"], self.previous.get(data["id"]))
            response = self.model.generate([{"role": "user", "content": prompt}])
            self.store.addUsage(response)

            if not response.ok or not response.text.strip():
                failed.append(data["id"])
                self.log.logMessage(f'API error on item {data["id"]}: {response.error}')
                continue

            self.store.add({
                "id": data["id"],
                "Question": data["question"],
                "Rewritten": response.text,
            })
            self.log.logMessage(f'改寫問題 (Rewritten)：\n{response.text}')

        self.store.save()
        return failed

    @staticmethod
    def getTokenLens(model: Model, data):
        """Calculate token usage for the rewritten text."""
        return model.getTokenLens(data.get("Rewritten", ""))
