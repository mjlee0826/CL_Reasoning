from Model.Model import Model
from Dataset.Dataset import Dataset
from Dataset.path import translatedBaseDir, rewrittenBaseDir, rewriteFileName
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Arm.GenerationRecord import GenerationRecord
from File.ResultStore import ResultStore
from Log.Log import Log

import json
import os
from tqdm import tqdm

SCHEMA_VERSION = "generation/v1"
NO_BASE_OUTPUT_REASON = ("The base arm's output is empty for these items, so no reflection call was made (the legacy "
                         "self-reflection run skipped them). Recorded as no answer (tokens 0); prompt_hash is the "
                         "prompt that would have been sent.")

class Generate(Strategy):
    """
    Runs one arm (ArmSpec) of one model over one dataset and writes GenerationRecords into a ResultStore
    (result/arms/{model}/{dataset}/{arm}.json).

    - Resume = repair: item_ids already in the store are skipped. Failed API calls are not written,
      so rerunning the same command fills them in.
    - Parse failures are kept as they are (parse_ok=False) and never re-sampled, so parse-failure rates
      stay comparable across arms.
    - A derived arm (F:{lang}, self-reflection) puts its base arm's output into the prompt, so it needs the
      complete base arm file (`baseStore`). Items whose base output is empty get no call and are written as
      outputs without an answer (listed in metadata no_answer_fill).
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, arm: ArmSpec, store: ResultStore,
                 baseStore: ResultStore | None = None):
        super().__init__(config)
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.arm: ArmSpec = arm
        self.store: ResultStore = store
        self.baseStore: ResultStore | None = baseStore
        self.promptBuilder = PromptBuilder(arm)

        self.config.displayName += f" ({arm.arm_id})"
        self.checkInputs()

    @staticmethod
    def questionSourcePath(arm: ArmSpec, dataset_type: str) -> str | None:
        """File the arm's question text is loaded from, or None for the original English question."""
        if arm.questionSource != "original":
            return os.path.join(rewrittenBaseDir, rewriteFileName(dataset_type, arm.rewriteVersion))
        if arm.lang != "en":
            return os.path.join(translatedBaseDir, f"{dataset_type}_{arm.language.capitalize()}.json")
        return None

    def checkInputs(self):
        dataset_config = self.dataset.config
        if dataset_config.sample != 1:
            raise ValueError("Generate requires sample == 1 (item_id must be unique); use S-axis seeds for repeats")
        uses_rewrite = self.arm.questionSource != "original"
        if dataset_config.language != self.arm.language or bool(dataset_config.useRewrite) != uses_rewrite \
                or (uses_rewrite and dataset_config.rewriteVersion != self.arm.rewriteVersion):
            raise ValueError(f"Dataset config does not match arm {self.arm.arm_id}; build it with ArmSpec.to_dataset_config")

        # Dataset silently falls back to English when the translation / rewrite file is missing
        path = self.questionSourcePath(self.arm, dataset_config.datasetType)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Question source for {self.arm.arm_id} not found: {path}")

        # ... and silently keeps the original question for any id the rewrite file does not cover
        if uses_rewrite:
            with open(path, encoding="utf-8") as f:
                covered = {r["id"] for r in json.load(f)[1:] if str(r.get("Rewritten", "")).strip()}
            missing = [data["id"] for data in self.dataset.getData() if data["id"] not in covered]
            if missing:
                raise ValueError(f"{path} does not cover {len(missing)} items (e.g. {missing[:5]}); finish run_rewrite.py first")

        # A resumed file must belong to the same run
        if self.store.metadata:
            self.checkRun(self.store, self.arm.arm_id)

        if self.arm.is_derived:
            self.checkBase()

    def checkRun(self, store: ResultStore, arm_id: str):
        """The file at `store` must hold arm_id of this model and dataset."""
        dataset_config = self.dataset.config
        meta = store.metadata
        expected = (arm_id, self.model.config.modelType, dataset_config.datasetType, dataset_config.nums)
        found = (meta.get("Arm", {}).get("arm_id"), meta.get("Model", {}).get("modelType"),
                 meta.get("Dataset", {}).get("datasetType"), meta.get("Dataset", {}).get("nums"))
        if expected != found:
            raise ValueError(f"{store.path} holds a different run: {found} != {expected}")

    def checkBase(self):
        """A derived arm's prompt contains its base arm's output, so the base arm file must be complete first."""
        base_id = self.arm.base_arm_id
        if self.baseStore is None or not self.baseStore.records:
            path = self.baseStore.path if self.baseStore else "(none given)"
            raise FileNotFoundError(f"{self.arm.arm_id} needs the outputs of {base_id}: {path} not found; generate {base_id} first")
        self.checkRun(self.baseStore, base_id)
        missing = [data["id"] for data in self.dataset.getData() if not self.baseStore.has(data["id"])]
        if missing:
            raise ValueError(f"{self.baseStore.path} misses {len(missing)} items (e.g. {missing[:5]}); finish {base_id} first")

    def messagesFor(self, data: dict) -> list[dict]:
        """Messages of one item; a derived arm's prompt contains the base arm's output for that item."""
        base_raw_text = self.baseStore.records[data["id"]]["raw_text"] if self.arm.is_derived else None
        return self.promptBuilder.messages(data["question"], base_raw_text)

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
            if self.arm.is_derived:
                self.store.metadata["base_arm_file"] = self.baseStore.path
        # False when the provider cannot take a seed (Gemini): the record's seed is then only a replicate id
        self.store.metadata["seed_sent_to_provider"] = self.arm.seed is not None and self.model.SUPPORTS_SEED

        todo =[data for data in self.dataset.getData() if not self.store.has(data["id"])]
        self.log.logMessage(f'{self.arm.arm_id}: {len(todo)} items to generate')

        failed = []
        pbar = tqdm(total=len(todo), desc=f"Generating {self.arm.arm_id}")
        for data in todo:
            messages = self.messagesFor(data)
            if self.arm.is_derived and not self.baseStore.records[data["id"]]["raw_text"]:
                self.addNoAnswer(data, messages)
                pbar.update()
                continue

            response = self.model.generate(messages, temperature=self.arm.temperature, seed=self.arm.seed)
            self.store.addUsage(response)

            if not response.ok:
                failed.append(data["id"])
                self.log.logMessage(f'API error on item {data["id"]}: {response.error}')
                pbar.update()
                continue

            if response.refused:
                # The provider blocked the content: that is the model's answer, kept as an output without an
                # answer (scored as wrong), the same way the legacy Gemini anchor recorded such an item
                self.store.metadata.setdefault("refusals", []).append(
                    {"item_id": data["id"], "reason": response.refusal_reason})
                self.log.logMessage(f'Item {data["id"]} refused: {response.refusal_reason}')

            record = self.buildRecord(data, messages, response)
            self.store.add(record.to_dict())

            self.log.logMessage(f'模型輸出 (Result)：\n{record.raw_text}')
            self.log.logMessage(f'My Answer: {record.parsed_answer} | Correct Answer: {record.gold}\n')
            pbar.update()

        pbar.close()
        self.store.save()
        return failed

    def addNoAnswer(self, data: dict, messages: list[dict]):
        """Derived arm, empty base output: no call; written as an output without an answer (scored as wrong)."""
        base = self.baseStore.records[data["id"]]
        fill = self.store.metadata.setdefault("no_answer_fill", {"ids": [], "reason": NO_BASE_OUTPUT_REASON})
        fill["ids"] = sorted(set(fill["ids"]) | {data["id"]})
        self.store.add(GenerationRecord(
            item_id=data["id"],
            arm_id=self.arm.arm_id,
            axis=self.arm.axis,
            is_anchor=self.arm.is_anchor,
            lang=self.arm.lang,
            raw_text="",
            parsed_answer="",
            parse_ok=False,
            tokens_in=0,
            tokens_out=0,
            model=self.model.config.modelType,
            model_version_string=base["model_version_string"],
            temperature=self.arm.temperature,
            seed=self.arm.seed,
            prompt_hash=PromptBuilder.promptHash(messages),
            gold=str(data.get("answer", "")),
        ).to_dict())
        self.log.logMessage(f'Item {data["id"]}: empty {self.arm.base_arm_id} output, recorded as no answer')
