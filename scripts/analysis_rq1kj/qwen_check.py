"""
qwen_check.py — RQ1-KJ 的離線核對：Qwen 在 M12 的 Judge 結果是不是流程造成的（不呼叫 API，不改任何現有檔案）

    一、五個數字（四個模型 × M12、M3L；只算不一致題，全部題目、不切分；最多票平手的題目排除並報題數）
        1. Judge 最終答案 = 最多票答案的比例
        2. 隨機挑一個候選時等於最多票答案的期望比例 = 平均（最多票的票數 / K）
        3. Judge 最終答案 = 位置 1 候選的答案的比例；位置 1 候選的答案 = 最多票答案的比例
        4. Judge 的正確率；m = 平均（答對的候選數 / K）
        5. Judge 輸出中提到的不同候選編號數（每題一次）的平均；K
    二、qwen × M12，「多數決對、Judge 錯」（不含平手題）中用 numpy.random.default_rng(0) 抽 20 題，逐題核對
        (a) 編號 → 候選 → 答案：presentation_order[choice−1] = chosen_arm；該 path 的 parsed_answer = final_answer；
            用題目原文 + 各候選 raw_text 依 presentation_order 重建 prompt，sha256 = 紀錄的 prompt_sha256
            （確認 prompt 裡的「Answer i」就是第 i 個位置的候選）
        (b) b1：輸出最後一行 {"choice": N} 的 N = 記錄的編號（流程）；
            b2：推理文字要選的候選與 JSON 是否矛盾（模型行為，人工閱讀，標註在 B2_NOTES；不觸發「流程問題」）
        (c) 沒有截斷（輸出 tokens < 8,000）、格式正常（有 Reasoning process / Final Choice、最後一行是唯一的 choice JSON）
    三、事先寫下的讀法 -> 結論；20 題全文放在報告最後

資料：result/analysis/rq1kj/judge_outputs（正式分析用的那一次；不讀 pilot_rep2/ 與 precheck/）、result/arms、
題目原文（Runner.builders.buildEnglishDataset，與 run_menu_judge.py 相同）。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq1kj/qwen_check.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
# 題目原文只讀本機的 HF 快取，不對 Hub 發請求
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd

from Aggregator.JudgeAggregator import JudgeAggregator
from Arm.PromptBuilder import PromptBuilder
from Strategy.PromptAbstractFactory.PromptJudgeChoiceFactory import PromptJudgeChoiceFactory
from Runner.builders import buildEnglishDataset
from Analysis.alignment import CellData
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import loadPathBlock, ordered
from Analysis.menuJudge import MODELS, DATASETS, MENUS, NUMS, OUT_DIR, JUDGE_DIR, loadMenuRecords
from Analysis.menuJudgeStats import judgeArrays, voteGroups

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
CHECK_MENUS = ["M12", "M3L"]
SAMPLE_MODEL, SAMPLE_MENU, SAMPLE_SIZE, SAMPLE_SEED = "qwen", "M12", 20, 0
READING_MARGIN = 5.0          # 百分點：數字 1 − 數字 2
CAP_TOKENS = 8000             # 輸出 tokens ≥ 8,000 視為寫到上限（max_tokens 8,192）
OUT_FILE = "qwen_check.md"

# 數字 5：「Answer N」「Answers 1, 3 and 5」「Answers 1-12 / 1 to 12 / 1 through 12」，不分大小寫，只算 1..K；
# 先拿掉輸出裡的 {"choice": N}（解析用的 JSON 不算提到）
MENTION = re.compile(r"\banswers?\s*#?\s*(\d+(?:\s*(?:,\s*(?:and\s+|or\s+|&\s*)?|and\s+|or\s+|&\s*|/\s*|-|–|—|\s+to\s+|\s+through\s+)"
                     r"\s*(?:answers?\s*)?#?\s*\d+)*)", re.IGNORECASE)
MENTION_TOKEN = re.compile(r"\d+|-|–|—|\bto\b|\bthrough\b", re.IGNORECASE)
RANGE_TOKENS = {"-", "–", "—", "to", "through"}
# b1 的獨立解析（不用流程裡的 CHOICE_PATTERN）：最後一個 "choice": N
B1_PATTERN = re.compile(r'"choice"\s*:\s*"?\s*(?:answer\s*)?(\d+)', re.IGNORECASE)
HEADING = lambda name: re.compile(rf"^[#*\s]*{name}[*:\s]*$", re.IGNORECASE | re.MULTILINE)

# ------------------------------------------------------------------
# b2：人工閱讀 20 題全文後的標註（2026-10-05，Claude）。key = (dataset, item_id)。
#   一致 = 推理文字要選的候選就是 JSON 的編號；矛盾 = 推理文字說要選別的、或說 JSON 選的那個是錯的；
#   未明說 = 推理文字沒有說要選哪一個
# ------------------------------------------------------------------
_ODD_ONE = "說 Answer {n} 是錯的（{why}），其餘答案正確；最後選 {n}"
B2_NOTES: dict[tuple[str, int], tuple[str, str]] = {
    ("mmlu", 400): ("矛盾", _ODD_ONE.format(n=6, why="把 price fixing (C) 當答案")),
    ("mmlu", 1019): ("一致", "說 Answer 1 的推導最正確，選 1"),
    ("mmlu", 2095): ("矛盾", _ODD_ONE.format(n=9, why="搞錯數字 6 的位數")),
    ("mmlu", 6954): ("矛盾", _ODD_ONE.format(n=8, why="選 Referent")),
    ("mathqa", 13): ("一致", "說 Answer 1 算出 191 是正確的，選 1"),
    ("mathqa", 306): ("一致", "自己推得 45 ounces，說 Answer 1、4、5、7、8 都得到 45，選列出的第一個 1。"
                              "但 Answer 1（S:T1.0:seed2）原文其實算出 60 ounces、選 d：文字與 JSON 一致，錯在對候選內容的描述"),
    ("truthfulqa", 556): ("矛盾", "說 Answer 6 錯（說 planchette 自己會動），其餘正確；接著寫「the most correct answer is 6」，選 6"),
    ("commonsenseqa", 42): ("一致", "論證 stressful (A) 更具體，說 answer 2 最好，選 2"),
    ("commonsenseqa", 93): ("矛盾", _ODD_ONE.format(n=5, why="選 nightmare (D)")),
    ("commonsenseqa", 266): ("矛盾", _ODD_ONE.format(n=7, why="選 fast food drive-thru (D)")),
    ("commonsenseqa", 344): ("一致", "說 Answer 1 的 sweating (C) 最正確，選 1"),
    ("commonsenseqa", 389): ("一致", "說 Answer 6 的 dog show (B) 正確，選 6"),
    ("commonsenseqa", 460): ("矛盾", "說 Answer 4 錯（選 hotel (C)），其餘正確；接著寫「the most correct answer is Answer 4」，選 4"),
    ("commonsenseqa", 834): ("矛盾", _ODD_ONE.format(n=7, why="說法官在定罪後進行審判")),
    ("commonsenseqa", 1170): ("矛盾", _ODD_ONE.format(n=4, why="選 hard (C)")),
    ("commonsenseqa", 1267): ("矛盾", _ODD_ONE.format(n=3, why="選 going down hill (B)")),
    ("commonsenseqa", 1614): ("一致", "說 answer 12 的 peculiar (C) 最貼近 quirky，選 12"),
    ("commonsenseqa", 1753): ("矛盾", _ODD_ONE.format(n=2, why="選 desk (D)")),
    ("commonsenseqa", 1847): ("矛盾", _ODD_ONE.format(n=6, why="選 counter (D)")),
    ("commonsenseqa", 1952): ("矛盾", _ODD_ONE.format(n=6, why="選 fight unfairly (B)")),
}


def parseArgs():
    parser = ArgumentParser(description="RQ1-KJ offline check: is Qwen's M12 judge result a pipeline artifact?")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--judge-dir", default=os.path.join(OUT_DIR, JUDGE_DIR))
    parser.add_argument("--out", default=os.path.join(OUT_DIR, OUT_FILE))
    return parser.parse_args()


def mentionedNumbers(text: str, k: int) -> set[int]:
    text = JudgeAggregator.CHOICE_PATTERN.sub(" ", text or "")
    found = set()
    for match in MENTION.finditer(text):
        tokens = [t.lower() for t in MENTION_TOKEN.findall(match.group(1))]
        previous, pending_range = None, False
        for token in tokens:
            if token in RANGE_TOKENS:
                pending_range = previous is not None
                continue
            number = int(token)
            if pending_range and previous < number:
                found.update(range(previous, number + 1))
            else:
                found.add(number)
            previous, pending_range = number, False
    return {n for n in found if 1 <= n <= k}


def plurality(answers: list[str], compare) -> tuple[str, int, bool]:
    """(最多票答案, 票數, 是否平手)。"""
    groups = voteGroups(answers, compare)
    top = max(count for _, count in groups)
    leaders = [answer for answer, count in groups if count == top]
    return leaders[0], top, len(leaders) > 1


# ------------------------------------------------------------------
# 一、五個數字
# ------------------------------------------------------------------
def itemRows(block, menu: str, records: dict) -> list[dict]:
    codes = ordered(MENUS[menu])
    judge = judgeArrays(block, codes, records, f"{block.model} | {block.dataset} | {menu}")
    k, compare, rows = len(codes), block.compare, []
    for i in np.flatnonzero(judge.judged):
        answers = [block.answers[c][i] for c in codes]
        top_answer, top_votes, tie = plurality(answers, compare)
        first = judge.order[i][0]
        record = records[int(block.item_ids[i])]
        rows.append({
            "model": block.model, "dataset": block.dataset, "menu": menu, "item_id": int(block.item_ids[i]), "K": k, "tie": tie,
            "plurality_answer": top_answer, "plurality_votes": top_votes,
            "judge_is_plurality": compare(judge.final[i], top_answer),
            "random_is_plurality": top_votes / k,
            "judge_is_first": compare(judge.final[i], block.answers[first][i]),
            "first_is_plurality": compare(block.answers[first][i], top_answer),
            "judge_correct": bool(judge.correct[i]),
            "m": float(np.mean([block.correct[c][i] for c in codes])),
            "vote_correct": compare(block.gold[i], top_answer),
            "mentioned": len(mentionedNumbers(record["trace"].get("judge_output"), k)),
        })
    return rows


def fiveNumbers(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for menu in CHECK_MENUS:
        for model in MODELS:
            group = items[(items.model == model) & (items.menu == menu)]
            kept = group[~group.tie]
            rows.append({
                "菜單": menu, "模型": MODEL_LABELS[model], "不一致題": len(group), "排除（平手）": int(group.tie.sum()), "納入": len(kept),
                "1. Judge = 最多票": 100 * kept.judge_is_plurality.mean(),
                "2. 隨機 = 最多票": 100 * kept.random_is_plurality.mean(),
                "1 − 2（pp）": 100 * (kept.judge_is_plurality.mean() - kept.random_is_plurality.mean()),
                "3. Judge = 位置 1": 100 * kept.judge_is_first.mean(),
                "位置 1 = 最多票": 100 * kept.first_is_plurality.mean(),
                "4. Judge 正確率": 100 * kept.judge_correct.mean(),
                "m": 100 * kept.m.mean(),
                "5. 提到的編號數": kept.mentioned.mean(),
                "K": int(kept.K.iloc[0]),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 二、逐題檢視
# ------------------------------------------------------------------
def samplePool(items: pd.DataFrame) -> pd.DataFrame:
    """qwen × M12、不含平手、多數決對而 Judge 錯；依資料集（DATASETS 的順序）再依 item_id 排序。"""
    pool = items[(items.model == SAMPLE_MODEL) & (items.menu == SAMPLE_MENU) & ~items.tie & items.vote_correct & ~items.judge_correct]
    pool = pool.assign(_d=pool.dataset.map(DATASETS.index)).sort_values(["_d", "item_id"]).drop(columns="_d")
    return pool.reset_index(drop=True)


def formatChecks(text: str, tokens_out: int, usage_out) -> dict:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    choices = JudgeAggregator.CHOICE_PATTERN.findall(text or "")
    out = max(tokens_out, usage_out or 0)
    checks = {
        "tokens_ok": out < CAP_TOKENS,
        "has_reasoning": bool(HEADING("Reasoning process").search(text or "")),
        "has_final": bool(HEADING("Final Choice").search(text or "")),
        "last_line_is_choice": bool(JudgeAggregator.CHOICE_PATTERN.fullmatch(last)),
        "one_choice": len(choices) == 1,
    }
    checks["c"] = all(checks.values())
    checks["c_note"] = "；".join(name for name, ok in checks.items() if name != "c" and not ok) or "—"
    return checks


def inspectItem(row, block, cell: CellData, records: dict, questions: dict) -> dict:
    record = records[row.item_id]
    trace = record["trace"]
    k = row.K
    position = int(np.flatnonzero(block.item_ids == row.item_id)[0])
    order = record["presentation_order"]
    arm_answers = {arm_id: cell.arm(arm_id)[row.item_id]["parsed_answer"] for arm_id in order}
    choice, chosen_arm = trace.get("choice"), trace.get("chosen_arm")
    # (a)
    permutation_ok = sorted(order) == sorted(PATHS[c] for c in MENUS[SAMPLE_MENU])
    mapped_ok = choice is not None and order[choice - 1] == chosen_arm
    answer_ok = chosen_arm is not None and arm_answers[chosen_arm] == record["final_answer"]
    prompt = PromptJudgeChoiceFactory().getPrompt("english", questions[row.item_id],
                                                  [cell.arm(arm_id)[row.item_id]["raw_text"] for arm_id in order])
    sha_ok = PromptBuilder.promptSha256([{"role": "user", "content": prompt}]) == trace.get("prompt_sha256")
    # (b1)
    b1_matches = B1_PATTERN.findall(trace.get("judge_output") or "")
    b1_number = int(b1_matches[-1]) if b1_matches else None
    b1_ok = b1_number is not None and b1_number == choice
    b2 = B2_NOTES.get((row.dataset, row.item_id), ("未標註", ""))
    fmt = formatChecks(trace.get("judge_output"), record["tokens_out"], (record.get("call") or {}).get("usage_out"))
    return {
        "dataset": row.dataset, "item_id": row.item_id, "K": k, "order": order, "arm_answers": arm_answers,
        "gold": block.gold[position], "plurality_answer": row.plurality_answer, "plurality_votes": row.plurality_votes,
        "choice": choice, "chosen_arm": chosen_arm, "final_answer": record["final_answer"], "output": trace.get("judge_output") or "",
        "tokens_out": record["tokens_out"], "usage_out": (record.get("call") or {}).get("usage_out"),
        "a_permutation": permutation_ok, "a_mapped": mapped_ok, "a_answer": answer_ok, "a_prompt_sha": sha_ok,
        "a": permutation_ok and mapped_ok and answer_ok and sha_ok,
        "b1_number": b1_number, "b1": b1_ok, "b2": b2[0], "b2_note": b2[1], **fmt,
    }


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def conclusion(inspected: list[dict], numbers: pd.DataFrame) -> list[str]:
    a_bad = [x for x in inspected if not x["a"]]
    b_bad = [x for x in inspected if not x["b1"]]
    row = numbers[(numbers["菜單"] == SAMPLE_MENU) & (numbers["模型"] == MODEL_LABELS[SAMPLE_MODEL])].iloc[0]
    gap = row["1 − 2（pp）"]
    b2 = pd.Series([x["b2"] for x in inspected]).value_counts()
    lines = [f"- (a) 不符 {len(a_bad)} 題、(b1) 不符 {len(b_bad)} 題、(c) 異常 {sum(not x['c'] for x in inspected)} 題"
             f"（共 {len(inspected)} 題）。",
             "- b2（模型行為，依事先決定不觸發「流程問題」）：" + "、".join(f"{label} {count} 題" for label, count in b2.items())
             + "。矛盾的題目，推理文字都說最後選中的那個候選是錯的、其餘（多數）是對的。",
             f"- Qwen × M12：數字 1 = {row['1. Judge = 最多票']:.1f}%，數字 2 = {row['2. 隨機 = 最多票']:.1f}%，"
             f"數字 1 − 數字 2 = {gap:+.1f}pp。"]
    if a_bad or b_bad:
        bad = sorted({(x["dataset"], x["item_id"]) for x in a_bad + b_bad})
        lines.append(f"- **讀法：(a) 或 (b) 有不符 → 流程問題。** 停下來回報，不修正、不重跑。不符的題目：{bad}")
    elif abs(gap) <= READING_MARGIN:
        lines.append("- **讀法：20 題都相符，且數字 1 與數字 2 相差在 5 個百分點以內 → Qwen 面對 12 個候選時接近隨機挑選，"
                     "是模型行為，RQ1-KJ 的數字照用。**")
    elif gap >= READING_MARGIN:
        lines.append("- **讀法：20 題都相符，但數字 1 比數字 2 高出 5 個百分點以上 → 只回報，不下結論。**")
    else:
        lines.append(f"- **數字 1 比數字 2 低 {-gap:.1f}pp，不在事先寫下的三種情況內 → 只回報，不下結論。**")
    return lines


def writeReport(path: str, args, items: pd.DataFrame, numbers: pd.DataFrame, pool: pd.DataFrame, picks: np.ndarray,
                inspected: list[dict], record_counts: dict):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    out = [f"# 核對一：Qwen 在 M12 的 Judge 結果是不是流程造成的", "",
           f"產生時間：{now}；程式：`scripts/analysis_rq1kj/qwen_check.py`。不呼叫 API，不修改任何現有檔案。", ""]

    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.judge_dir}/{{模型}}/{{資料集}}/{{M12,M3L}}.json`：正式分析用的那一次（試跑第一次的紀錄就在其中，"
            "全量沿用）。不讀 `pilot_rep2/`（試跑第二次）與 `precheck/`。"
            "欄位：`presentation_order`、`final_answer`、`tokens_out`、`trace.judge_output`、`trace.choice`、`trace.chosen_arm`、"
            "`trace.prompt_sha256`、`call.usage_out`。",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/` 的 M12 12 個 path 檔：`parsed_answer`、`parse_ok`、`gold`（經 "
            "`Analysis.menuVote.loadPathBlock`，與 RQ1-KJ 分析相同），以及 20 題的 `raw_text`（重建 prompt）。",
            "- 題目原文：`Runner.builders.buildEnglishDataset`（與 `run_menu_judge.py` 相同的來源；只讀本機 HF 快取，離線模式），"
            "只用於 (a) 重建 20 題的 prompt。",
            "- 讀入的 Judge 紀錄數（每檔都剛好等於該菜單子集內的不一致題，`judgeArrays` 核對）：",
            ""]
    out += [table(pd.DataFrame([{"菜單": m, **{MODEL_LABELS[x]: record_counts[(x, m)] for x in MODELS}} for m in CHECK_MENUS]),
                  floatfmt=".0f"), ""]

    out += ["## (2) 五個數字", "",
            "只算不一致題（菜單內每條 path 都有答案、答案不完全相同），四個資料集合併、全部題目、不切分。"
            "最多票平手（最高票數由兩個以上的答案共有）的題目排除，五個數字都在排除後的題目上算。百分比（%）。",
            "",
            "- 1：Judge 最終答案 = 最多票答案。2：每題「最多票的票數 ÷ K」的平均。",
            "- 3：Judge 最終答案 = 位置 1 候選的答案；「位置 1 = 最多票」是位置 1 候選的答案剛好是最多票答案的比例。",
            "- 4：Judge 在這些題目的正確率（無效選擇算錯）；m = 每題「答對的候選數 ÷ K」的平均。",
            "- 5：Judge 輸出中提到的不同候選編號數的平均。算法：「Answer N」、「Answers 1, 3 and 5」的列舉、"
            "「Answers 1-12 / 1 to 12 / 1 through 12」的範圍展開；不分大小寫；只算 1..K；先拿掉最後的 `{\"choice\": N}`。",
            ""]
    out += [table(numbers), ""]

    n_ok = sum(x["a"] for x in inspected), sum(x["b1"] for x in inspected), sum(x["c"] for x in inspected)
    b2_counts = pd.Series([x["b2"] for x in inspected]).value_counts()
    out += ["## (3) 20 題的檢視結果", "",
            f"題目池：Qwen × M12、不含平手題、多數決（= 唯一的最多票答案）對而 Judge 錯，共 {len(pool)} 題；"
            "依資料集（mmlu、mathqa、truthfulqa、commonsenseqa）再依 item_id 排序，"
            f"`numpy.random.default_rng({SAMPLE_SEED}).choice({len(pool)}, {SAMPLE_SIZE}, replace=False)`，"
            "下表依題目池的順序列出（「抽中順序」是 choice 回傳的順序）。",
            "",
            "- (a) 編號 → 候選 → 答案：presentation_order 是 12 條 path 的排列；presentation_order[編號 − 1] = 記錄的 chosen_arm；"
            "該 path 在 `result/arms` 的 parsed_answer = 記錄的 final_answer；用題目原文與各候選 raw_text 依 presentation_order "
            "重建 prompt，sha256 = 記錄的 prompt_sha256（確認 prompt 裡的「Answer i」就是第 i 個位置的候選）。四項都成立才算 ✓。",
            "- (b) b1：輸出裡最後一個 `\"choice\": N` 的 N（獨立的 regex，不用流程的解析式）= 記錄的編號；並由人工閱讀確認最後一行。"
            "b2：推理文字要選的候選和 JSON 的編號是否矛盾（人工閱讀）。依事先決定，只有 b1 不符算流程問題；b2 是模型行為，只計數。",
            "- (c) 輸出 tokens（紀錄與 API 取大者）< 8,000；有「Reasoning process」與「Final Choice」標題；最後一個非空行就是 "
            "choice JSON；整段輸出只有一個 choice JSON。",
            ""]
    rows = []
    for rank, x in enumerate(inspected, start=1):
        rows.append({
            "#": rank, "抽中順序": x["draw"] + 1, "資料集": x["dataset"], "item_id": x["item_id"],
            "編號": x["choice"], "對應 path": x["chosen_arm"], "對應答案": x["final_answer"],
            "最多票（票數）": f"{x['plurality_answer']}（{x['plurality_votes']}）", "正確答案": x["gold"],
            "(a)": mark(x["a"]), "(b1)": mark(x["b1"]), "(b2)": x["b2"], "(c)": mark(x["c"]),
            "tokens": x["usage_out"] if x["usage_out"] is not None else x["tokens_out"],
        })
    out += [table(pd.DataFrame(rows), floatfmt=".0f"), ""]
    out += [f"統計：(a) 相符 {n_ok[0]} / {len(inspected)}；(b1) 相符 {n_ok[1]} / {len(inspected)}；(c) 正常 {n_ok[2]} / {len(inspected)}；"
            "(b2) " + "、".join(f"{label} {count}" for label, count in b2_counts.items()) + "。", ""]
    if inspected:
        out += ["逐題備註（b2 的閱讀摘要，以及 (a)、(b1)、(c) 不符的項目）：", ""]
        for rank, x in enumerate(inspected, start=1):
            notes = []
            if not x["a"]:
                notes.append("(a) " + "、".join(name for name in ("a_permutation", "a_mapped", "a_answer", "a_prompt_sha") if not x[name]))
            if not x["b1"]:
                notes.append(f"(b1) 輸出最後的 choice = {x['b1_number']}，記錄 = {x['choice']}")
            if not x["c"]:
                notes.append(f"(c) {x['c_note']}")
            if x["b2_note"]:
                notes.append(f"(b2 {x['b2']}) {x['b2_note']}")
            out.append(f"- #{rank} {x['dataset']} {x['item_id']}：" + "；".join(notes))
        out.append("")

    out += ["## (4) 對照讀法的結論", ""] + conclusion(inspected, numbers) + [""]

    out += ["## (5) 20 題全文", ""]
    for rank, x in enumerate(inspected, start=1):
        out += [f"### #{rank}　Qwen3-8B · {x['dataset']} · item {x['item_id']}", "",
                f"解析出的編號 {x['choice']} → path `{x['chosen_arm']}` → 答案 `{x['final_answer']}`；正確答案 `{x['gold']}`；"
                f"最多票 `{x['plurality_answer']}`（{x['plurality_votes']} / {x['K']} 票）。"
                f"(a) {mark(x['a'])}　(b1) {mark(x['b1'])}　(b2) {x['b2']}　(c) {mark(x['c'])}"
                f"（輸出 tokens：紀錄 {x['tokens_out']}、API {x['usage_out']}）", ""]
        order_rows = [{"位置": i, "path": arm_id, "解析答案": x["arm_answers"][arm_id],
                       "答對": "✓" if x["arm_answers"][arm_id] == x["gold"] else ""} for i, arm_id in enumerate(x["order"], start=1)]
        out += ["presentation_order：", "", table(pd.DataFrame(order_rows), floatfmt=".0f"), "", "Judge 原始輸出：", "",
                fence(x["output"]), ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")


def main():
    args = parseArgs()
    items, blocks, records_by, record_counts = [], {}, {}, {}
    for model in MODELS:
        for dataset in DATASETS:
            block = loadPathBlock(args.armdir, model, dataset, args.aggdir)
            blocks[(model, dataset)] = block
            for menu in CHECK_MENUS:
                _, records = loadMenuRecords(args.judge_dir, model, dataset, menu)
                records_by[(model, dataset, menu)] = records
                record_counts[(model, menu)] = record_counts.get((model, menu), 0) + len(records)
                items += itemRows(block, menu, records)
    items = pd.DataFrame(items)
    numbers = fiveNumbers(items)

    pool = samplePool(items)
    picks = np.random.default_rng(SAMPLE_SEED).choice(len(pool), SAMPLE_SIZE, replace=False)
    questions, inspected = {}, []
    for draw, index in sorted(enumerate(picks), key=lambda pair: pair[1]):
        row = pool.iloc[index]
        if row.dataset not in questions:
            questions[row.dataset] = {data["id"]: data["question"] for data in buildEnglishDataset(row.dataset, NUMS).getData()}
        cell = CellData(args.armdir, args.aggdir, SAMPLE_MODEL, row.dataset)
        x = inspectItem(row, blocks[(SAMPLE_MODEL, row.dataset)], cell, records_by[(SAMPLE_MODEL, row.dataset, SAMPLE_MENU)],
                        questions[row.dataset])
        x["draw"] = draw
        inspected.append(x)

    writeReport(args.out, args, items, numbers, pool, picks, inspected, record_counts)
    print(numbers.to_string(index=False))
    print(f"(a) {sum(x['a'] for x in inspected)}/20  (b1) {sum(x['b1'] for x in inspected)}/20  (c) {sum(x['c'] for x in inspected)}/20")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
