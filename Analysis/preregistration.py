import hashlib
import re

# ------------------------------------------------------------------
# 事先登記的判定標準檔（rq1_criteria.md、rq2_criteria.md）：
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
