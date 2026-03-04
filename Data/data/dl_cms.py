from datasets import load_dataset

dataset = load_dataset("tau/commonsense_qa")
dataset["train"].to_json("./data/commenseqa.json")

import json

# 載入 JSONL
with open("data/commenseqa.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 存成 JSON Array 格式（排版）
with open("data/commenseqa.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
