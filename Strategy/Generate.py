from Model.Model import Model
from Dataset.Dataset import Dataset
from Dataset.path import translatedBaseDir, rewrittenBaseDir
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Arm.GenerationRecord import GenerationRecord
from File.ResultStore import ResultStore
from Log.Log import Log

import os
from tqdm import tqdm

SCHEMA_VERSION = "generation/v1"

class Generate(Strategy):
    """
    Runs one arm (ArmSpec) of one model over one dataset and writes GenerationRecords into a ResultStore
    (result/arms/{model}/{dataset}/{arm}.json).

    - Resume = repair: item_ids already in the store are skipped. Failed API calls are not written,
      so rerunning the same command fills them in.
    - Parse failures are kept as they are (parse_ok=False) and never re-sampled, so parse-failure rates
      stay comparable across arms.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, arm: ArmSpec, store: ResultStore):
        super().__init__(config)
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.arm: ArmSpec = arm
        self.store: ResultStore = store
        self.promptBuilder = PromptBuilder(arm)

        self.config.displayName += f" ({arm.arm_id})"
        self.checkInputs()

    @staticmethod
    def questionSourcePath(arm: ArmSpec, dataset_type: str) -> str | None:
        """File the arm's question text is loaded from, or None for the original English question."""
        if arm.questionSource == "rewrite":
            return os.path.join(rewrittenBaseDir, f"{dataset_type}_english.json")
        if arm.lang != "en":
            return os.path.join(translatedBaseDir, f"{dataset_type}_{arm.language.capitalize()}.json")
        return None

    def checkInputs(self):
        dataset_config = self.dataset.config
        if dataset_config.sample != 1:
            raise ValueError("Generate requires sample == 1 (item_id must be unique); use S-axis seeds for repeats")
        if dataset_config.language != self.arm.language or bool(dataset_config.useRewrite) != (self.arm.questionSource == "rewrite"):
            raise ValueError(f"Dataset config does not match arm {self.arm.arm_id}; build it with ArmSpec.to_dataset_config")

        # Dataset silently falls back to English when the translation / rewrite file is missing
        path = self.questionSourcePath(self.arm, dataset_config.datasetType)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Question source for {self.arm.arm_id} not found: {path}")

        # A resumed file must belong to the same run
        meta = self.store.metadata
        if meta:
            expected = (self.arm.arm_id, self.model.config.modelType, dataset_config.datasetType, dataset_config.nums)
            found = (meta.get("Arm", {}).get("arm_id"), meta.get("Model", {}).get("modelType"),
                     meta.get("Dataset", {}).get("datasetType"), meta.get("Dataset", {}).get("nums"))
            if expected != found:
                raise ValueError(f"{self.store.path} holds a different run: {found} != {expected}")

    def buildRecord(self, data: dict, messages: list[dict], response) -> GenerationRecord:
        parsed = self.parseAnswer(response.text)
        return GenerationRecord(
            item_id=data["id"],
            arm_id=self.arm.arm_id,
            axis=self.arm.axis,
            is_anchor=self.arm.is_anchor,
            lang=self.arm.lang,
            raw_text=response.text,
            parsed_answer=parsed,
            parse_ok=GenerationRecord.isParseOk(parsed),
            tokens_in=sum(self.model.countTokens(m["content"]) for m in messages),
            tokens_out=self.model.countTokens(response.text),
            model=self.model.config.modelType,
            model_version_string=response.model_version,
            temperature=self.arm.temperature,
            seed=self.arm.seed,
            prompt_hash=PromptBuilder.promptHash(messages),
            gold=str(data.get("answer", "")),
        )

    def getRes(self) -> list:
        """
        Generates every missing item. Returns the item_ids whose API call failed (still missing).
        """
        self.log.logInfo(self, self.model, self.dataset)

        if not self.store.metadata:
            self.store.metadata = {
                "Model": self.model.config.to_dict(),
                "Dataset": self.dataset.config.to_dict(),
                "Strategy": self.config.to_dict(),
                "Arm": self.arm.to_dict(),
                "schema_version": SCHEMA_VERSION,
                "source": "generate",
            }
        # False when the provider cannot take a seed (Gemini): the record's seed is then only a replicate id
        self.store.metadata["seed_sent_to_provider"] = self.arm.seed is not None and self.model.SUPPORTS_SEED

        todo =[data for data in self.dataset.getData() if not self.store.has(data["id"])]
        self.log.logMessage(f'{self.arm.arm_id}: {len(todo)} items to generate')

        failed = []
        pbar = tqdm(total=len(todo), desc=f"Generating {self.arm.arm_id}")
        for data in todo:
            messages = self.promptBuilder.messages(data["question"])
            response = self.model.generate(messages, temperature=self.arm.temperature, seed=self.arm.seed)
            self.store.addUsage(response)

            if not response.ok:
                failed.append(data["id"])
                self.log.logMessage(f'API error on item {data["id"]}: {response.error}')
                pbar.update()
                continue

            record = self.buildRecord(data, messages, response)
            self.store.add(record.to_dict())

            self.log.logMessage(f'模型輸出 (Result)：\n{record.raw_text}')
            self.log.logMessage(f'My Answer: {record.parsed_answer} | Correct Answer: {record.gold}\n')
            pbar.update()

        pbar.close()
        self.store.save()
        return failed

    @staticmethod
    def getTokenLens(model: Model, data):
        """Output tokens are already stored on the record."""
        return data.get("tokens_out") or 0
