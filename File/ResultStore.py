import json
import os

class ResultStore():
    """
    Resumable writer for [metadata, record, ...] result files (result/arms, result/aggregations).

    - Loads the existing file if present, so strategies can skip item_ids that are already done.
    - save() writes a temp file and os.replace()s it, so an interrupted run never leaves a truncated JSON.
    - Records are keyed by item_id; File.File can read the output directly.
    """
    def __init__(self, path: str, key: str = "item_id", checkpoint_every: int = 50):
        self.path = path
        self.key = key
        self.checkpoint_every = checkpoint_every
        self.metadata: dict = {}
        self.records: dict = {}
        self._unsaved = 0

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw:
                self.metadata = raw[0]
                self.records = {record[key]: record for record in raw[1:]}

    def has(self, item_id) -> bool:
        return item_id in self.records

    def add(self, record: dict):
        """Adds a record and checkpoints every `checkpoint_every` new records."""
        self.records[record[self.key]] = record
        self._unsaved += 1
        if self._unsaved >= self.checkpoint_every:
            self.save()

    def addUsage(self, response):
        """Accumulates provider-reported usage of an LLMResponse into metadata['api_usage'] (reconciliation only)."""
        usage = self.metadata.setdefault("api_usage", {"calls": 0, "calls_without_usage": 0,
                                                       "prompt_tokens": 0, "completion_tokens": 0})
        usage["calls"] += 1
        if response.usage_in is None or response.usage_out is None:
            usage["calls_without_usage"] += 1
            return
        usage["prompt_tokens"] += response.usage_in
        usage["completion_tokens"] += response.usage_out

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        full_data = [self.metadata] + [self.records[k] for k in sorted(self.records)]
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, self.path)
        self._unsaved = 0
