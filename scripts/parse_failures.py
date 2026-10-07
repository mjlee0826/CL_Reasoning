"""
parse_failures.py — 離線核對：M12 的 12 條 path 裡，沒有答案的輸出是什麼原因（不呼叫 API，不改任何現有檔案）

    一、每個區塊（4 模型 × 4 資料集）一列
        1. 總題數；12 條 path 中至少一條沒有答案的題數與比例
        2. 沒有答案的輸出總數（path × 題目）
        3. 其中「寫到 token 上限」：輸出 tokens ≥ 8,000（arm 紀錄沒有 finish_reason，只能用這個條件）
        4. 其中「有重複迴圈」：有一段 50 個字元的片段，在輸出中不重疊地出現 5 次以上
        5. 其餘（沒有寫到上限）依序分類：輸出為空或被供應商擋下 / 有輸出但格式不符（沒有 {"…": "…"} 的答案 JSON）/
           其他（有答案 JSON，但值是空白、None 或 null）
    二、deepseek4.1flash × mathqa：依 path 列出 2–5，以及每條 path 的平均輸出 tokens（全部題目；排除 ≥ 8,000 的輸出）
    三、事先寫下的讀法：寫到上限的比例 ≥ 90% / < 90%（從不是寫到上限的輸出用 default_rng(0) 抽 10 題附原文）；
        任何區塊「格式不符」> 該區塊總輸出數的 1% 就標出

「沒有答案」= parse_ok 為 False（解析結果是空的、"null" 或 "None"），與 RQ1-K / RQ1-KJ 保留題目時相同。
資料：result/arms/{模型}/{資料集}/ 的 M12 12 個 path 檔（不含 F_en、F_zh）。

用法（從 repo root）：
    conda run -n clreasoning python scripts/parse_failures.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import numpy as np
import pandas as pd

from Analysis.alignment import CellData, loadRecords
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import MODELS, DATASETS, MENUS

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
CODES = MENUS["M12"]
CAP_TOKENS = 8000                     # max_tokens 是 8,192（四個模型相同）
LOOP_LENGTH, LOOP_TIMES = 50, 5       # 字元、次數
FOCUS = ("deepseek4.1flash", "mathqa")
CAP_SHARE_THRESHOLD = 0.90
FORMAT_FLAG = 0.01                    # 格式不符 / 該區塊總輸出數
SAMPLE_SIZE, SAMPLE_SEED = 10, 0
# 任何鍵的答案 JSON（解析器 Strategy.parseAnswer 找的是 {"…": "…"}；這裡也接受空字串與 null，用來分出「其他」）
ANSWER_JSON = re.compile(r'\{\s*"[^"]+"\s*:\s*(?:"[^"]*"|null)\s*\}')

EMPTY, FORMAT, OTHER = "empty_or_refused", "format_mismatch", "other"


def parseArgs():
    parser = ArgumentParser(description="Offline check: why M12 path outputs have no parsed answer")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--out", default="result/analysis/parse_failures.md")
    return parser.parse_args()


def hasLoop(text: str, length: int = LOOP_LENGTH, times: int = LOOP_TIMES) -> bool:
    """某個長度 `length` 的片段不重疊地出現 ≥ `times` 次（更長的重複片段一定含一個重複的 `length` 字元片段）。"""
    if len(text) < length * times:
        return False
    seen = {}   # 片段 -> (不重疊的次數, 下一次可計入的起點)
    for i in range(len(text) - length + 1):
        window = text[i:i + length]
        count, next_start = seen.get(window, (0, 0))
        if i >= next_start:
            count += 1
            if count >= times:
                return True
            seen[window] = (count, i + length)
    return False


def classify(record: dict, refused: set) -> dict:
    text = record.get("raw_text") or ""
    capped = record["tokens_out"] >= CAP_TOKENS
    if capped:
        reason = None
    elif not text.strip() or record["item_id"] in refused:
        reason = EMPTY
    elif not ANSWER_JSON.search(text):
        reason = FORMAT
    else:
        reason = OTHER
    return {"capped": capped, "loop": hasLoop(text), "reason": reason, "refused": record["item_id"] in refused}


def loadOutputs(armdir: str, aggdir: str) -> tuple[pd.DataFrame, dict]:
    """每個 (模型, 資料集, path, 題目) 一列：是否有答案、輸出 tokens；沒有答案的另加原因。"""
    rows, texts = [], {}
    for model in MODELS:
        for dataset in DATASETS:
            cell = CellData(armdir, aggdir, model, dataset)
            for code in CODES:
                meta, records = loadRecords(cell.armFile(PATHS[code]))
                refused = {r["item_id"] for r in meta.get("refusals") or []}
                for item_id, record in records.items():
                    row = {"model": model, "dataset": dataset, "path": code, "item_id": item_id,
                           "answered": bool(record["parse_ok"]), "tokens_out": record["tokens_out"]}
                    if not row["answered"]:
                        row.update(classify(record, refused))
                        texts[(model, dataset, code, item_id)] = record.get("raw_text") or ""
                    rows.append(row)
    return pd.DataFrame(rows), texts


def summarize(group: pd.DataFrame) -> dict:
    """沒有答案的輸出：數量、寫到上限、迴圈、其餘的原因。"""
    missing = group[~group.answered]
    n = len(missing)
    rest = missing[~missing.capped.astype(bool)] if n else missing
    share = lambda k: k / n if n else float("nan")
    return {
        "n_missing": n,
        "n_capped": int(missing.capped.sum()) if n else 0, "capped_share": share(int(missing.capped.sum()) if n else 0),
        "n_loop": int(missing.loop.sum()) if n else 0, "loop_share": share(int(missing.loop.sum()) if n else 0),
        "n_capped_loop": int((missing.capped & missing.loop).sum()) if n else 0,
        "n_empty": int((rest.reason == EMPTY).sum()) if n else 0, "n_refused": int(rest.refused.sum()) if n else 0,
        "n_format": int((rest.reason == FORMAT).sum()) if n else 0, "n_other": int((rest.reason == OTHER).sum()) if n else 0,
    }


def blockTable(outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            group = outputs[(outputs.model == model) & (outputs.dataset == dataset)]
            n_items = group.item_id.nunique()
            any_missing = group.groupby("item_id").answered.all().eq(False).sum()
            s = summarize(group)
            rows.append({"model": model, "dataset": dataset, "n_items": n_items, "n_outputs": len(group),
                         "n_items_missing": int(any_missing), "items_missing_share": any_missing / n_items, **s,
                         "format_share_of_outputs": s["n_format"] / len(group)})
    return pd.DataFrame(rows)


def pathTable(outputs: pd.DataFrame) -> pd.DataFrame:
    model, dataset = FOCUS
    rows = []
    for code in CODES:
        group = outputs[(outputs.model == model) & (outputs.dataset == dataset) & (outputs.path == code)]
        under = group[group.tokens_out < CAP_TOKENS]
        rows.append({"path": code, "arm": PATHS[code], **summarize(group),
                     "tokens_mean_all": group.tokens_out.mean(), "tokens_mean_under_cap": under.tokens_out.mean(),
                     "n_outputs_ge_cap": int((group.tokens_out >= CAP_TOKENS).sum()),
                     "n_answered_ge_cap": int((group.answered & (group.tokens_out >= CAP_TOKENS)).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}%"


def fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def writeReport(path: str, args, outputs: pd.DataFrame, blocks: pd.DataFrame, paths: pd.DataFrame, texts: dict):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    out = ["# 核對二：沒有答案的輸出是什麼原因", "",
           f"產生時間：{now}；程式：`scripts/parse_failures.py`。不呼叫 API，不修改任何現有檔案。", ""]

    out += ["## (1) 讀了哪些檔案與欄位", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/` 的 M12 12 個 path 檔（"
            + "、".join(f"`{PATHS[c].replace(':', '_')}.json`" for c in CODES) + "），4 模型（"
            + "、".join(MODELS) + "）× 4 資料集，共 192 個檔；不含 F_en、F_zh，也不含 result/arms/deepseek（舊的 V3.2）"
            "與 result/archive（Gemini 2.5）。",
            "- 逐筆欄位：`parse_ok`（False = 沒有答案，即解析結果是空的、\"null\" 或 \"None\"，與 RQ1-K / RQ1-KJ 保留題目時相同）、"
            "`raw_text`、`tokens_out`、`item_id`；檔頭的 `refusals`（供應商擋下的題目）。",
            "- **沒有 `finish_reason` 欄位**：arm 紀錄不存它，檔頭的 `api_usage` 只有整檔總數。所以「寫到 token 上限」只用"
            f"「輸出 tokens ≥ {CAP_TOKENS:,}」一個條件；finish_reason = length 的數量無法列出。"
            "`tokens_out` 是用各模型的 tokenizer 對輸出文字重算的數（GPT / Qwen 的舊結果匯入時重算；DeepSeek 與 API 回報相同；"
            "Gemini 用 count_tokens，與 API 差幾個 token）。四個模型的 max_tokens 都是 8,192。",
            f"- 「有重複迴圈」：輸出中某一段 {LOOP_LENGTH} 個字元的片段，不重疊地出現 {LOOP_TIMES} 次以上"
            f"（更長的重複片段一定含一個重複的 {LOOP_LENGTH} 字元片段，所以只查 {LOOP_LENGTH} 字元）。只對沒有答案的輸出計算。",
            "- 原因分類：先看是否寫到上限；其餘（沒有寫到上限）依序判斷 ——",
            "  - 輸出為空或被供應商擋下：`raw_text` 是空的，或題目列在檔頭的 `refusals`；",
            "  - 有輸出但格式不符：輸出裡沒有 `{\"…\": \"…\"}` 形式的答案 JSON（解析器 `Strategy.parseAnswer` 找的就是它）；",
            "  - 其他：有答案 JSON，但值是空白、\"None\" 或 null（模型表明選項中沒有答案）。這不是解析規則的問題。",
            ""]

    table = []
    for r in blocks.itertuples():
        flag = " ⚠" if r.format_share_of_outputs > FORMAT_FLAG else ""
        table.append({
            "模型": MODEL_LABELS[r.model], "資料集": r.dataset, "總題數": r.n_items,
            "≥1 條沒答案（題）": f"{r.n_items_missing}（{pct(r.items_missing_share)}）",
            "沒答案的輸出": r.n_missing,
            "寫到上限（≥8,000）": f"{r.n_capped}（{pct(r.capped_share)}）",
            "finish_reason=length": "無此欄位",
            "有迴圈": f"{r.n_loop}（{pct(r.loop_share)}）",
            "其餘：空 / 擋下": f"{r.n_empty}（擋下 {r.n_refused}）",
            "其餘：格式不符": f"{r.n_format}（佔總輸出 {100 * r.format_share_of_outputs:.2f}%）{flag}",
            "其餘：其他": r.n_other,
        })
    total = summarize(outputs)
    out += ["## (2) 16 個區塊", "",
            "每個區塊的總輸出數 = 12 × 總題數。括號內：「≥1 條沒答案」是佔總題數；「寫到上限」「有迴圈」是佔該區塊沒答案的輸出；"
            "「格式不符」是佔該區塊總輸出數（> 1% 標 ⚠）。",
            "", pd.DataFrame(table).to_markdown(index=False), "",
            f"合計：沒答案的輸出 {total['n_missing']}；寫到上限 {total['n_capped']}（{pct(total['capped_share'])}）；"
            f"有迴圈 {total['n_loop']}（{pct(total['loop_share'])}）；寫到上限且有迴圈 {total['n_capped_loop']}；"
            f"其餘：空 / 擋下 {total['n_empty']}（擋下 {total['n_refused']}）、格式不符 {total['n_format']}、其他 {total['n_other']}。", ""]

    model, dataset = FOCUS
    focus = blocks[(blocks.model == model) & (blocks.dataset == dataset)].iloc[0]
    rows = []
    for r in paths.itertuples():
        rows.append({
            "path": f"{r.path}（{r.arm}）", "沒答案的輸出": r.n_missing,
            "寫到上限": f"{r.n_capped}（{pct(r.capped_share)}）", "有迴圈": f"{r.n_loop}（{pct(r.loop_share)}）",
            "空 / 擋下": r.n_empty, "格式不符": r.n_format, "其他": r.n_other,
            "平均輸出 tokens（全部）": f"{r.tokens_mean_all:.1f}",
            "平均輸出 tokens（排除 ≥8,000）": f"{r.tokens_mean_under_cap:.1f}",
            "≥8,000 的輸出（其中有答案）": f"{r.n_outputs_ge_cap}（{r.n_answered_ge_cap}）",
        })
    out += [f"## (3) {MODEL_LABELS[model]} × {dataset} 的細分", "",
            f"總題數 {focus.n_items}；至少一條 path 沒答案 {focus.n_items_missing} 題（{pct(focus.items_missing_share)}），"
            f"也就是 M12 保留 {pct(1 - focus.items_missing_share)}。沒答案的輸出 {focus.n_missing}，"
            f"其中寫到上限 {focus.n_capped}（{pct(focus.capped_share)}）、有迴圈 {focus.n_loop}（{pct(focus.loop_share)}）、"
            f"兩者都是 {focus.n_capped_loop}。",
            "", "平均輸出 tokens 的「全部」是該 path 的全部題目（含有答案的）；「排除 ≥8,000」拿掉所有輸出 tokens ≥ 8,000 的輸出"
            "（不論有沒有答案，最後一欄列出被拿掉的數量與其中有答案的數量）。",
            "", pd.DataFrame(rows).to_markdown(index=False), ""]

    out += ["## (4) 對照讀法的結論", ""]
    share = focus.capped_share
    if share >= CAP_SHARE_THRESHOLD:
        out += [f"- {MODEL_LABELS[model]} × {dataset} 沒答案的輸出中，寫到上限的比例 = {pct(share)}（{focus.n_capped} / {focus.n_missing}）"
                f"≥ 90% → 附錄寫「幾乎全部是重複迴圈寫到上限」，並附這個比例。"
                f"（同一批輸出中，有迴圈的比例 = {pct(focus.loop_share)}。）"]
        sample_section = []
    else:
        out += [f"- {MODEL_LABELS[model]} × {dataset} 沒答案的輸出中，寫到上限的比例 = {pct(share)} < 90% → 照實列出各原因的比例"
                f"（第 (3) 節），並從不是寫到上限的輸出中抽 {SAMPLE_SIZE} 題附在最後。"]
        pool = outputs[(outputs.model == model) & (outputs.dataset == dataset) & ~outputs.answered & ~outputs.capped.astype(bool)]
        pool = pool.assign(_p=pool.path.map(CODES.index)).sort_values(["_p", "item_id"]).reset_index(drop=True)
        picks = np.random.default_rng(SAMPLE_SEED).choice(len(pool), min(SAMPLE_SIZE, len(pool)), replace=False)
        sample_section = ["## 附：不是寫到上限的輸出（抽樣）", "",
                          f"題目池 {len(pool)} 筆（依 path 的 M12 順序再依 item_id 排序），"
                          f"`numpy.random.default_rng({SAMPLE_SEED}).choice({len(pool)}, {len(picks)}, replace=False)`。", ""]
        for index in sorted(picks):
            r = pool.iloc[index]
            sample_section += [f"### {r.path} · item {r.item_id}（原因：{r.reason}；tokens {r.tokens_out}）", "",
                               fence(texts[(model, dataset, r.path, r.item_id)]), ""]
    flagged = blocks[blocks.format_share_of_outputs > FORMAT_FLAG]
    if len(flagged):
        out += ["- **「有輸出但格式不符」超過該區塊總輸出數 1% 的區塊（可能是解析規則的問題）：** "
                + "、".join(f"{MODEL_LABELS[r.model]} × {r.dataset}（{100 * r.format_share_of_outputs:.2f}%）" for r in flagged.itertuples())]
    else:
        worst = blocks.loc[blocks.format_share_of_outputs.idxmax()]
        out += [f"- 沒有任何區塊的「有輸出但格式不符」超過該區塊總輸出數的 1%"
                f"（最高：{MODEL_LABELS[worst.model]} × {worst.dataset}，{worst.n_format} 筆，{100 * worst.format_share_of_outputs:.2f}%）。"]
    out += [""] + sample_section
    Path(path).write_text("\n".join(out), encoding="utf-8")


def main():
    args = parseArgs()
    outputs, texts = loadOutputs(args.armdir, args.aggdir)
    blocks, paths = blockTable(outputs), pathTable(outputs)
    writeReport(args.out, args, outputs, blocks, paths, texts)
    print(blocks[["model", "dataset", "n_items_missing", "n_missing", "n_capped", "n_loop", "n_empty", "n_format", "n_other"]]
          .to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
