import hashlib
import json
import os
import re

# ------------------------------------------------------------------
# 事先登記的判定標準檔（rq1_criteria.md、rq2_criteria.md、rq1k_criteria.md、rq1kj_criteria.md）：
#   「確認：」行填上時間之前，分析不得執行；確認之後檔案不得修改，每份輸出記錄它的 sha256
# ------------------------------------------------------------------


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def confirmationLine(criteria_path: str) -> str | None:
    """判定標準檔的「確認：」行；尚未填入確認時間時回傳 None。"""
    with open(criteria_path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\s*-\s*確認：(.*)", line)
            if match:
                text = match.group(1).strip()
                return None if "待填" in text or not text else text
    return None


def loadStepFile(path: str, step: str, criteria_sha256: str, script: str) -> dict:
    """執行前檢查的結果檔（precheck.json、pilot.json …）；檔案不存在或寫於另一份判定標準下就停。"""
    if not os.path.exists(path):
        raise SystemExit(f"❌ {path} not found: run the `{step}` step of {script} first")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("criteria_sha256") != criteria_sha256:
        raise SystemExit(f"❌ {path} was written under another criteria file (sha256 {data.get('criteria_sha256')})")
    return data
