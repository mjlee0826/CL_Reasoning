"""
dissenter_check.py — RQ1-KJ 的離線核對三：Judge 選到「少數那一個」的比例（不呼叫 API，不改任何判定與現有檔案）

    一、M12 依最多票的票數 top 分組（11、10、9、8、≤7）：題數；Judge = 最多票；隨機 = 最多票（top / 12）；
        Judge 選中的候選是唯一持有者（隨機期望 = 唯一持有者人數 / 12）；Judge 與多數決的正確率。合併與分資料集各一張表
    二、M3L、M3S、M3P：2 比 1 的題目上 Judge 選到少數的比例（隨機 1/3）
    三、M12 的文字輔助計數（粗略規則，只當參考）
    四、gpt4omini、deepseek4.1flash、gemini3.1flashlite 的 M12：top ≥ 9、多數決對、選中唯一持有者的題目，
        各用 default_rng(0) 抽 10 題，人工標記（D4_NOTES）
    五、假設性的重新計分（只做 qwen M12，不取代事先登記的結果）：top ≥ 9 且選中唯一持有者的題目改記為最多票答案，
        A_J、Excess_J 用 RQ1-KJ 相同的切分與 S_in 重算
    六、事先寫下的讀法

定義：top = 最多票答案的票數；最多票平手的題目排除（第五節的分母除外）。唯一持有者 = 答案在 K 個候選中只有它一個人持有。
無效選擇（沒有有效編號）留在分母，算「沒有選到唯一持有者 / 少數」、算錯；第三節沒有 N 可比，不進分母。

資料：result/analysis/rq1kj/judge_outputs（正式分析用的那一次；不讀 pilot_rep2/ 與 precheck/）、result/arms、
result/analysis/rq1kj/rq1kj_blocks.csv（第五節先核對原始 A_J、Excess_J 可重現）。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq1kj/dissenter_check.py
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Aggregator.JudgeAggregator import JudgeAggregator
from Analysis.blockStats import summarize
from Analysis.menuVote import loadPathBlock, ordered
from Analysis.menuJudge import MODELS, DATASETS, MENUS, MENU_NAMES, MAIN_MENU, TOLERANCE, OUT_DIR, JUDGE_DIR, loadMenuRecords
from Analysis.menuJudgeStats import judgeArrays, evaluateBlockMenu, voteGroups, blocksWithSplits

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
THREE_MENUS = ["M3L", "M3S", "M3P"]
TOP_GROUPS = ["11", "10", "9", "8", "≤7"]
FOCUS_MODEL = "qwen"
OTHER_MODELS = [m for m in MODELS if m != FOCUS_MODEL]
HIGH_TOP = 9                    # 第四、五節：top ≥ 9
SAMPLE_SIZE, SAMPLE_SEED = 10, 0
RATIO, SOLE_LIMIT = 3.0, 0.15   # 讀法：qwen ≥ 3 × 其他每一個模型；其他模型 < 15%
OUT_FILE = "dissenter_check.md"

# 第三節：「Answer N」與「Answers 2, 6 and 9」的列舉算提到；「Answers 1-12 / 1 to 12」的範圍不算（範圍兩端也不算）。
# 先拿掉 {"choice": N}
MENTION = re.compile(r"\banswers?\s*#?\s*(\d+(?:\s*(?:,\s*(?:and\s+|or\s+|&\s*)?|and\s+|or\s+|&\s*|/\s*|-|–|—|\s+to\s+|\s+through\s+)"
                     r"\s*(?:answers?\s*)?#?\s*\d+)*)", re.IGNORECASE)
MENTION_TOKEN = re.compile(r"\d+|-|–|—|\bto\b|\bthrough\b", re.IGNORECASE)
RANGE_TOKENS = {"-", "–", "—", "to", "through"}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
NEGATIVE = re.compile(r"\b(?:incorrect|incorrectly|wrong|errors?|mistakes?|flawed)\b", re.IGNORECASE)
OTHERS = re.compile(r"\b(?:all other answers|the other answers|other answers|all others|the others|the rest|"
                    r"(?:the )?remaining answers|most (?:of the )?answers)\b", re.IGNORECASE)
CORRECT = re.compile(r"\bcorrect(?:ly)?\b", re.IGNORECASE)

# ------------------------------------------------------------------
# 第四節：人工閱讀全文後的標記（2026-10-05，Claude）。key = (model, dataset, item_id)
#   矛盾 = 推理文字說選中的候選是錯的，卻輸出它的編號；一致 = 推理文字主張選中的候選是對的；其他 = 說明情況
# ------------------------------------------------------------------
_LETTER = "多數答案是 {a}，輸出的是 1"
_MISLISTED = "文字要選的是多數答案，把 Answer {n} 列在多數（正確）的一方並輸出 {n}；但 Answer {n} 原文主張的是 {x}。" + _LETTER
D4_NOTES: dict[tuple[str, str, int], tuple[str, str]] = {
    # GPT-4o mini
    ("gpt4omini", "mmlu", 1128): ("其他", "文字說 Answer 1「不含任何整數、沒有錯把 9 算進去」所以最正確，選 1；但 Answer 1 原文答 3、5、7、9、11（C），"
                                         "正是文字批評的「含 9」。對選中候選的描述與原文不符"),
    ("gpt4omini", "mmlu", 1207): ("其他", _MISLISTED.format(n=1, x="B", a="A")),
    ("gpt4omini", "mmlu", 6320): ("其他", "文字說 Answer 2 的推理有誤（其實 Answer 2 答 A，是多數），多數答案 A 最正確，輸出 1；"
                                         "Answer 1 原文主張 C。" + _LETTER.format(a="A")),
    ("gpt4omini", "mmlu", 6573): ("一致", "說 Answer 3（選 B）最符合 MacKinnon 的論點，選 3"),
    ("gpt4omini", "mathqa", 516): ("其他", _MISLISTED.format(n=1, x="567 → 最接近的 b", a="a")),
    ("gpt4omini", "mathqa", 582): ("其他", _MISLISTED.format(n=1, x="e（都不對）", a="a")),
    ("gpt4omini", "mathqa", 1154): ("矛盾", "說第一個答案錯誤地選 b) 5:8，正確的是選 a) 8:5 的答案，輸出 1。" + _LETTER.format(a="a")),
    ("gpt4omini", "truthfulqa", 377): ("其他", "文字主張選項 A（不用等）最正確，沒有提到 Answer 1，輸出 1；Answer 1 原文主張 B（等 30 分鐘）。"
                                              + _LETTER.format(a="A")),
    ("gpt4omini", "commonsenseqa", 103): ("其他", "文字主張多數答案 A，說選 D 的答案較不相關，沒有提到 Answer 1，輸出 1；Answer 1 原文主張 D。"
                                                 + _LETTER.format(a="A")),
    ("gpt4omini", "commonsenseqa", 603): ("其他", _MISLISTED.format(n=1, x="D", a="A")),
    # DeepSeek V4.1 Flash
    ("deepseek4.1flash", "mmlu", 2469): ("矛盾", "說 Answer 5 錯誤地選 C，其餘都正確選 D，「Answer 5 is the only one with a flawed final choice」，選 5"),
    ("deepseek4.1flash", "mmlu", 4234): ("一致", "說 Answer 6 指出題意的歧義、推理最嚴謹，選 6"),
    ("deepseek4.1flash", "mmlu", 9436): ("矛盾", "逐一檢查 12 個答案，「Only Answer 6 is incorrect」，選 6"),
    ("deepseek4.1flash", "mmlu", 13175): ("矛盾", "逐一檢查，「Only Answer 4 selects C, which is incorrect」，其餘正確，選 4"),
    ("deepseek4.1flash", "mathqa", 1960): ("一致", "自己算出不在選項中的值，說只有 Answer 8 選 e) none 是對的，選 8"),
    ("deepseek4.1flash", "truthfulqa", 411): ("其他", "反覆權衡後認為沒有一個答案正確（多數選的 A 不是地點，Answer 3 選的 C 事實錯誤），"
                                                     "把 Answer 3 當作「最不錯」的而選 3"),
    ("deepseek4.1flash", "commonsenseqa", 34): ("矛盾", "說 Answer 3 錯、是唯一的離群者，其他 11 個都正確，選 3"),
    ("deepseek4.1flash", "commonsenseqa", 780): ("矛盾", "說 Answer 6 錯、是唯一的離群者，其他都正確，選 6"),
    ("deepseek4.1flash", "commonsenseqa", 1210): ("矛盾", "說 Answer 3 錯、是唯一的離群者，其他都正確，選 3"),
    ("deepseek4.1flash", "commonsenseqa", 1746): ("一致", "論證 kitchen (B) 比 cupboard 更普遍正確，說 answer 10 最平衡，選 10"),
    # Gemini 3.1 Flash-Lite
    ("gemini3.1flashlite", "mmlu", 7302): ("一致", "說 Answer 2 對「與父母共有單倍型的機率」的推理最正確，選 2"),
    ("gemini3.1flashlite", "mathqa", 545): ("一致", "說 254.47 不等於 254，所以 Answer 1 選 e) none 才對，其餘錯，選 1"),
    ("gemini3.1flashlite", "mathqa", 1305): ("一致", "主張 (a, b) 與 (b, a) 算不同的組，只有 Answer 11 對，選 11"),
    ("gemini3.1flashlite", "commonsenseqa", 13): ("一致", "把題目當成文字遊戲，說只有 Answer 1 看出來，選 1"),
    ("gemini3.1flashlite", "commonsenseqa", 62): ("一致", "說題目出自特定小說，只有 Answer 5 抓到出處，選 5"),
    ("gemini3.1flashlite", "commonsenseqa", 603): ("一致", "說題目是特定的冷知識，只有 Answer 10 正確，選 10"),
    ("gemini3.1flashlite", "commonsenseqa", 1395): ("一致", "說題目問的是「奇怪」之處，只有 Answer 7（medical building）抓到，選 7"),
    ("gemini3.1flashlite", "commonsenseqa", 1533): ("一致", "把題目當成謎語，說只有 Answer 3 看出謎語的邏輯，選 3"),
    ("gemini3.1flashlite", "commonsenseqa", 1628): ("一致", "說這題在 CommonsenseQA 資料集的標準答案是 B，只有 Answer 9 看出來，選 9"),
    ("gemini3.1flashlite", "commonsenseqa", 1969): ("一致", "把題目當成文字遊戲，說只有 Answer 1 看出來，選 1"),
}


def parseArgs():
    parser = ArgumentParser(description="RQ1-KJ offline check 3: how often the judge picks the lone dissenter")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--judge-dir", default=os.path.join(OUT_DIR, JUDGE_DIR))
    parser.add_argument("--blocks", default=os.path.join(OUT_DIR, "rq1kj_blocks.csv"))
    parser.add_argument("--out", default=os.path.join(OUT_DIR, OUT_FILE))
    return parser.parse_args()


def listedNumbers(text: str) -> set[int]:
    """「Answer N」與列舉中的編號；範圍（1-12）的兩端不算。"""
    found = set()
    for match in MENTION.finditer(text):
        tokens = [t.lower() for t in MENTION_TOKEN.findall(match.group(1))]
        for j, token in enumerate(tokens):
            if token in RANGE_TOKENS:
                continue
            next_is_range = j + 1 < len(tokens) and tokens[j + 1] in RANGE_TOKENS
            previous_is_range = j > 0 and tokens[j - 1] in RANGE_TOKENS
            if not (next_is_range or previous_is_range):
                found.add(int(token))
    return found


def textCounts(output: str, choice: int) -> tuple[bool, bool]:
    """(選中的 N 和負面用詞在同一句, 輸出中有「其他答案正確」的句子)。"""
    text = JudgeAggregator.CHOICE_PATTERN.sub(" ", output or "")
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    negative = any(choice in listedNumbers(s) and NEGATIVE.search(s) for s in sentences)
    others = any(OTHERS.search(s) and CORRECT.search(s) for s in sentences)
    return negative, others


def topGroup(top: int) -> str:
    return str(top) if top >= 8 else "≤7"


# ------------------------------------------------------------------
# 逐題
# ------------------------------------------------------------------
def itemRows(block, menu: str, judge, records: dict) -> list[dict]:
    codes, compare, rows = ordered(MENUS[menu]), block.compare, []
    k = len(codes)
    for i in np.flatnonzero(judge.judged):
        answers = {c: block.answers[c][i] for c in codes}
        groups = voteGroups(list(answers.values()), compare)
        counts = sorted((count for _, count in groups), reverse=True)
        top = counts[0]
        tie = counts.count(top) > 1
        plurality = next(answer for answer, count in groups if count == top)
        held = {c: next(count for answer, count in groups if compare(answer, answers[c])) for c in codes}
        chosen = judge.chosen[i]
        record = records[int(block.item_ids[i])]
        rows.append({
            "model": block.model, "dataset": block.dataset, "menu": menu, "item_id": int(block.item_ids[i]), "K": k,
            "top": top, "tie": tie, "distribution": "-".join(map(str, counts)),
            "groups": " / ".join(f"{answer}×{count}" for answer, count in sorted(groups, key=lambda g: -g[1])),
            "plurality": plurality, "gold": block.gold[i],
            "choice": judge.choice[i], "chosen": chosen, "chosen_answer": answers[chosen] if chosen else "",
            "valid": chosen is not None,
            "chosen_sole": chosen is not None and held[chosen] == 1,
            "n_sole": sum(count == 1 for count in held.values()),
            "judge_is_plurality": compare(judge.final[i], plurality), "judge_correct": bool(judge.correct[i]),
            "vote_correct": compare(block.gold[i], plurality),
            "order": record["presentation_order"], "output": (record.get("trace") or {}).get("judge_output") or "",
            "row": int(i),
        })
    return rows


# ------------------------------------------------------------------
# 第一、二、三節
# ------------------------------------------------------------------
def groupStats(group: pd.DataFrame) -> dict:
    return {
        "題數": len(group),
        "2. Judge = 最多票": 100 * group.judge_is_plurality.mean(),
        "3. 隨機 = 最多票": 100 * (group.top / group.K).mean(),
        "4. 選中唯一持有者": 100 * group.chosen_sole.mean(),
        "隨機期望": 100 * (group.n_sole / group.K).mean(),
        "5. Judge 正確率": 100 * group.judge_correct.mean(),
        "多數決正確率": 100 * group.vote_correct.mean(),
    }


def sectionOne(items: pd.DataFrame, by_dataset: bool) -> pd.DataFrame:
    m12 = items[(items.menu == MAIN_MENU) & ~items.tie]
    keys = ["model", "dataset"] if by_dataset else ["model"]
    rows = []
    for model in MODELS:
        for dataset in (DATASETS if by_dataset else [None]):
            scope = m12[m12.model == model] if dataset is None else m12[(m12.model == model) & (m12.dataset == dataset)]
            for name in TOP_GROUPS:
                group = scope[scope.top.map(topGroup) == name]
                row = {"模型": MODEL_LABELS[model]} | ({"資料集": dataset} if by_dataset else {}) | {"top": name}
                rows.append(row | (groupStats(group) if len(group) else {"題數": 0}))
    return pd.DataFrame(rows)


def tieCounts(items: pd.DataFrame, menu: str) -> dict:
    scope = items[items.menu == menu]
    return {model: (int(scope[scope.model == model].tie.sum()), int((scope.model == model).sum())) for model in MODELS}


def sectionTwo(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for menu in THREE_MENUS:
        for model in MODELS:
            scope = items[(items.menu == menu) & (items.model == model)]
            two_one = scope[scope.distribution == "2-1"]
            rows.append({"菜單": menu, "模型": MODEL_LABELS[model], "不一致題": len(scope),
                         "1-1-1（排除）": int((scope.distribution == "1-1-1").sum()), "2 比 1": len(two_one),
                         "無效選擇": int((~two_one.valid).sum()),
                         "選到少數": 100 * two_one.chosen_sole.mean(), "隨機期望": 100 / 3,
                         "高於 1/3": "⚠" if two_one.chosen_sole.mean() > 1 / 3 else ""})
    return pd.DataFrame(rows)


def sectionThree(items: pd.DataFrame) -> pd.DataFrame:
    m12 = items[(items.menu == MAIN_MENU) & ~items.tie]
    rows = []
    for model in MODELS:
        scope = m12[m12.model == model]
        valid = scope[scope.valid]
        counts = [textCounts(output, int(choice)) for output, choice in zip(valid.output, valid.choice)]
        negative = np.array([c[0] for c in counts], dtype=bool)
        others = np.array([c[1] for c in counts], dtype=bool)
        rows.append({"模型": MODEL_LABELS[model], "不含平手的不一致題": len(scope), "無效選擇（不計）": int((~scope.valid).sum()),
                     "分母": len(valid), "1. N 與負面用詞同句": int(negative.sum()), "1. 比例": 100 * negative.mean(),
                     "2. 其中有「其他答案正確」": int((negative & others).sum()),
                     "2. 比例": 100 * (negative & others).sum() / negative.sum() if negative.any() else float("nan")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 第四節
# ------------------------------------------------------------------
def samplePools(items: pd.DataFrame) -> dict:
    m12 = items[(items.menu == MAIN_MENU) & ~items.tie & (items.top >= HIGH_TOP) & items.vote_correct & items.chosen_sole]
    pools = {}
    for model in OTHER_MODELS:
        pool = m12[m12.model == model]
        pool = pool.assign(_d=pool.dataset.map(DATASETS.index)).sort_values(["_d", "item_id"]).drop(columns="_d").reset_index(drop=True)
        if len(pool) > SAMPLE_SIZE:
            picks = np.random.default_rng(SAMPLE_SEED).choice(len(pool), SAMPLE_SIZE, replace=False)
        else:
            picks = np.arange(len(pool))
        pools[model] = (pool, sorted(int(p) for p in picks))
    return pools


# ------------------------------------------------------------------
# 第五節
# ------------------------------------------------------------------
def sectionFive(items: pd.DataFrame, armdir: str, aggdir: str, judge_dir: str, reference: pd.DataFrame) -> dict:
    m12 = items[items.menu == MAIN_MENU]
    rows, max_error, recoded_n = [], 0.0, 0
    for block, splits in blocksWithSplits(armdir, aggdir):
        codes = ordered(MENUS[MAIN_MENU])
        _, records = loadMenuRecords(judge_dir, block.model, block.dataset, MAIN_MENU)
        judge = judgeArrays(block, codes, records, f"{block.model} | {block.dataset} | {MAIN_MENU}")
        original = evaluateBlockMenu(block, MAIN_MENU, splits, judge)["row"]
        ref = reference[(reference.model == block.model) & (reference.dataset == block.dataset) & (reference.menu == MAIN_MENU)].iloc[0]
        max_error = max(max_error, abs(original["A_J"] - ref["A_J"]), abs(original["excess_J"] - ref["excess_J"]) / 100)
        row = {"model": block.model, "dataset": block.dataset, "A_J": original["A_J"], "excess_J": original["excess_J"],
               "n_judged": int(judge.judged.sum()), "n_recoded": 0, "A_J_recoded": original["A_J"], "excess_J_recoded": original["excess_J"]}
        if block.model == FOCUS_MODEL:
            mine = m12[(m12.model == block.model) & (m12.dataset == block.dataset)]
            target = mine[(mine.top >= HIGH_TOP) & mine.chosen_sole]
            final, correct = list(judge.final), judge.correct.copy()
            for r in target.itertuples():
                final[r.row] = r.plurality
                correct[r.row] = block.compare(block.gold[r.row], r.plurality)
            recoded = evaluateBlockMenu(block, MAIN_MENU, splits, replace(judge, final=final, correct=correct))["row"]
            row.update(n_recoded=len(target), A_J_recoded=recoded["A_J"], excess_J_recoded=recoded["excess_J"])
            recoded_n += len(target)
        rows.append(row)
    blocks = pd.DataFrame(rows)
    n_focus = int(blocks[blocks.model == FOCUS_MODEL].n_judged.sum())
    return {"blocks": blocks, "max_error": max_error, "n_recoded": recoded_n, "n_focus": n_focus,
            "original": summarize(blocks.excess_J), "recoded": summarize(blocks.excess_J_recoded)}


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def md(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def interval(s: dict) -> str:
    return f"{s['mean']:+.2f} [{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]，{s['n_positive']}/{s['n_blocks']} 為正"


def readings(one: pd.DataFrame, two: pd.DataFrame, pools: dict) -> list[str]:
    top11 = one[one.top == "11"].set_index("模型")["4. 選中唯一持有者"]
    focus = top11[MODEL_LABELS[FOCUS_MODEL]]
    others = {m: top11[MODEL_LABELS[m]] for m in OTHER_MODELS}
    ratios = {m: focus / v if v > 0 else float("inf") for m, v in others.items()}
    lines = ["top = 11 的第 4 項（選中唯一持有者）：" + f"Qwen3-8B {focus:.1f}%；"
             + "；".join(f"{MODEL_LABELS[m]} {v:.1f}%（Qwen 是它的 {ratios[m]:.1f} 倍）" for m, v in others.items()) + "。", ""]
    all_ratio = all(r >= RATIO for r in ratios.values())
    all_low = all(v < 100 * SOLE_LIMIT for v in others.values())
    high = [m for m, v in others.items() if v >= 100 * SOLE_LIMIT]
    lines.append(f"- 讀法一（Qwen ≥ 其他每一個模型的 3 倍，且其他三個都 < 15%）：倍數{'都' if all_ratio else '沒有都'} ≥ 3，"
                 f"其他三個{'都' if all_low else '沒有都'} < 15% → "
                 + ("**成立：這是 qwen 特有的輸出問題。**" if all_ratio and all_low else "不成立。"))
    lines.append("- 讀法二（其他任何一個模型在 top = 11 的第 4 項 ≥ 15%）："
                 + (f"**⚠ {', '.join(MODEL_LABELS[m] for m in high)} 達到 15% 以上：choice-k-v1 的 prompt 對不只一個模型有問題。**"
                    if high else "沒有，其他三個模型都低於 15%。"))
    flagged4 = []
    for model, (pool, picks) in pools.items():
        labels = [D4_NOTES.get((model, pool.iloc[p].dataset, int(pool.iloc[p].item_id)), ("未標記", ""))[0] for p in picks]
        if picks and labels.count("矛盾") * 2 >= len(picks):
            flagged4.append(f"{MODEL_LABELS[model]}（{labels.count('矛盾')} / {len(picks)}）")
    lines.append("- 讀法三（第四節其他模型的「矛盾」≥ 抽樣題數的一半）："
                 + (f"**⚠ {'、'.join(flagged4)}。**" if flagged4 else "沒有任何模型達到一半。"))
    over = two[two["選到少數"] > 100 / 3]
    lines.append("- 讀法四（三條 path 的菜單上，選到少數的比例高於 1/3）："
                 + (f"**⚠ " + "、".join(f"{r['模型']} × {r['菜單']}（{r['選到少數']:.1f}%）" for _, r in over.iterrows()) + "。**"
                    if len(over) else "沒有任何模型 × 菜單高於 1/3。"))
    return lines


def writeReport(args, items: pd.DataFrame, one: pd.DataFrame, one_ds: pd.DataFrame, two: pd.DataFrame, three: pd.DataFrame,
                pools: dict, five: dict):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    counts = items.groupby(["menu", "model"]).size()
    out = ["# 核對三：Judge 選到「少數那一個」的比例", "",
           f"產生時間：{now}；程式：`scripts/analysis_rq1kj/dissenter_check.py`。不呼叫 API，不改任何判定、不取代事先登記的計分、"
           "不修改現有檔案。", ""]

    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.judge_dir}/{{模型}}/{{資料集}}/{{M12,M3L,M3S,M3P}}.json`：正式分析用的那一次（試跑第一次的紀錄就在其中）。"
            "不讀 `pilot_rep2/`（試跑第二次）與 `precheck/`。欄位：`presentation_order`、`final_answer`、`trace.choice`、"
            "`trace.chosen_arm`、`trace.judge_output`。每檔的紀錄剛好等於該菜單子集內的不一致題（`judgeArrays` 核對）。",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：`parsed_answer`、`parse_ok`、`gold`，經 `Analysis.menuVote.loadPathBlock`"
            "（與 RQ1-KJ 分析相同；它會讀全部 14 個 path 檔，這裡只用菜單內的 path）。",
            f"- `{args.blocks}`：只在第五節用來核對原始 A_J、Excess_J 可以重現。",
            "- 讀入的 Judge 紀錄數（不一致題）：", "",
            md(pd.DataFrame([{"菜單": m, **{MODEL_LABELS[x]: int(counts[(m, x)]) for x in MODELS}} for m in MENU_NAMES]), ".0f"), "",
            "定義：top = 該題最多票答案的票數；最多票平手（最高票數由兩個以上的答案共有）的題目在第一到第四節排除。"
            "唯一持有者 = 答案在 K 個候選中只有它一個人持有。無效選擇（沒有有效編號）留在分母，算「沒有選到唯一持有者 / 少數」、"
            "正確率算錯（與事先登記的計分相同）；第三節沒有 N 可比，不進分母。", ""]

    ties = tieCounts(items, MAIN_MENU)
    out += ["## (2) 第一節：M12 依 top 分組", "",
            "M12 排除的平手題：" + "；".join(f"{MODEL_LABELS[m]} {t} / {n}" for m, (t, n) in ties.items()) + "。",
            "", "- 2：Judge 最終答案 = 最多票答案。3：隨機挑一個候選 = 最多票答案的期望（top ÷ 12 的平均）。",
            "- 4：Judge 選中的候選是唯一持有者；「隨機期望」= 每題唯一持有者人數 ÷ 12 的平均。",
            "- 5：Judge 的正確率與多數決（= 唯一的最多票答案）的正確率。百分比（%）。", "",
            "### 四個資料集合併", "", md(one), "", "### 依資料集", "", md(one_ds), ""]

    out += ["## (3) 第二節：三條 path 的菜單（2 比 1 的題目）", "",
            "「選到少數」= Judge 選中的候選就是 2 比 1 中那個少數（唯一持有者）。1-1-1 的題目沒有最多票，排除。百分比（%）。", "",
            md(two), ""]

    out += ["## (4) 第三節：文字上的輔助計數（粗略規則，只當參考）", "",
            "M12、不含平手的不一致題中，有有效編號 N 的題目。規則：",
            "- 先拿掉輸出中的 `{\"choice\": N}`；在「. ! ? 後接空白」與換行處切句。",
            "- 提到 N：句中有「Answer N」或列舉「Answers 2, 6 and 9」含 N（不分大小寫）；「Answers 1-12 / 1 to 12」的範圍不算，範圍兩端也不算。",
            "- 1：有一句同時提到 N 且含 incorrect、incorrectly、wrong、error(s)、mistake(s)、flawed 之一（整字、不分大小寫）。",
            "- 2：符合 1 的輸出中，另有一句含 all other answers、the other answers、other answers、all others、the others、the rest、"
            "(the) remaining answers、most (of the) answers 之一，並含 correct 或 correctly（整字，所以 incorrect 不算）。",
            "- 這個規則抓不到否定句與指代（例如「it is wrong」），也會把「Answer N correctly identifies the error」算進來。",
            "", md(three), ""]

    out += ["## (5) 第四節：其他三個模型的逐題檢視（標記統計）", "",
            f"題目池：M12、top ≥ {HIGH_TOP}、多數決對、Judge 選中的候選是唯一持有者；依資料集（mmlu、mathqa、truthfulqa、commonsenseqa）"
            f"再依 item_id 排序；每個模型各自用 `numpy.random.default_rng({SAMPLE_SEED}).choice(題目池大小, {SAMPLE_SIZE}, replace=False)`"
            "（不足 10 題就全部列出）。標記是人工閱讀全文後的判斷。", ""]
    rows = []
    for model, (pool, picks) in pools.items():
        labels = [D4_NOTES.get((model, pool.iloc[p].dataset, int(pool.iloc[p].item_id)), ("未標記", ""))[0] for p in picks]
        rows.append({"模型": MODEL_LABELS[model], "題目池": len(pool), "抽出": len(picks), "矛盾": labels.count("矛盾"),
                     "一致": labels.count("一致"), "其他": labels.count("其他"), "未標記": labels.count("未標記")})
    out += [md(pd.DataFrame(rows), ".0f"), ""]
    for model, (pool, picks) in pools.items():
        out += [f"- {MODEL_LABELS[model]}："]
        for rank, p in enumerate(picks, start=1):
            r = pool.iloc[p]
            label, note = D4_NOTES.get((model, r.dataset, int(r.item_id)), ("未標記", ""))
            out.append(f"  - #{rank} {r.dataset} {r.item_id}：{label}" + (f"。{note}" if note else ""))
    out.append("")

    b = five["blocks"]
    focus = b[b.model == FOCUS_MODEL]
    out += ["## (6) 第五節：假設性的數字", "",
            "> **這是假設性的重新計分，只用來評估規模；不取代事先登記的結果（rq1kj_criteria.md），不寫進任何判定。**", "",
            f"只改 Qwen3-8B 的 M12：top ≥ {HIGH_TOP} 且 Judge 選中的候選是唯一持有者的題目，最終答案改記為最多票答案；"
            "其他 12 個區塊不變。切分（makeSplits(n, 200, 0)）與 S_in 的算法和 RQ1-KJ 相同（`evaluateBlockMenu`）。",
            f"先用原始的 Judge 結果重算 16 個區塊：A_J、Excess_J 與 `rq1kj_blocks.csv` 的最大誤差 {five['max_error']:.1e}"
            f"（門檻 {TOLERANCE:.0e}）{'，可以重現' if five['max_error'] <= TOLERANCE else '，**無法重現，以下數字不可用**'}。", "",
            f"- 被改記的題數：{five['n_recoded']}，占 Qwen3-8B M12 不一致題（{five['n_focus']} 題，含平手題）的 "
            f"{100 * five['n_recoded'] / five['n_focus']:.1f}%。", ""]
    out += [md(pd.DataFrame([{"資料集": r.dataset, "不一致題": r.n_judged, "改記題數": r.n_recoded,
                              "A_J（原始）": 100 * r.A_J, "A_J（改記）": 100 * r.A_J_recoded,
                              "Excess_J（原始，pp）": r.excess_J, "Excess_J（改記，pp）": r.excess_J_recoded}
                             for r in focus.itertuples()]), ".2f"), "",
            f"- 16 個區塊的 Excess_J 平均與 95% t 區間（pp）：原始 {interval(five['original'])}；"
            f"只改記 Qwen 四個區塊後 {interval(five['recoded'])}。", ""]

    out += ["## (7) 對照讀法的結論", ""] + readings(one, two, pools) + [""]

    out += ["## (8) 第四節各題全文", ""]
    for model, (pool, picks) in pools.items():
        for rank, p in enumerate(picks, start=1):
            r = pool.iloc[p]
            label, note = D4_NOTES.get((model, r.dataset, int(r.item_id)), ("未標記", ""))
            order = "、".join(f"{i}={arm}" for i, arm in enumerate(r.order, start=1))
            out += [f"### {MODEL_LABELS[model]} #{rank} · {r.dataset} · item {r.item_id}　【{label}】", "",
                    f"票數分布：{r.groups}（正確答案 `{r.gold}`）。選中編號 {int(r.choice)} → path `{r.chosen}` → 答案 `{r.chosen_answer}`。",
                    f"presentation_order：{order}。" + (f"\n\n標記說明：{note}" if note else ""), "", fence(r.output), ""]
    Path(args.out).write_text("\n".join(out), encoding="utf-8")


def main():
    args = parseArgs()
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            block = loadPathBlock(args.armdir, model, dataset, args.aggdir)
            for menu in MENU_NAMES:
                _, records = loadMenuRecords(args.judge_dir, model, dataset, menu)
                judge = judgeArrays(block, ordered(MENUS[menu]), records, f"{model} | {dataset} | {menu}")
                rows += itemRows(block, menu, judge, records)
    items = pd.DataFrame(rows)
    one, one_ds, two, three = sectionOne(items, False), sectionOne(items, True), sectionTwo(items), sectionThree(items)
    pools = samplePools(items)
    five = sectionFive(items, args.armdir, args.aggdir, args.judge_dir, pd.read_csv(args.blocks))
    writeReport(args, items, one, one_ds, two, three, pools, five)
    print(one.to_string(index=False))
    print(two.to_string(index=False))
    print(three.to_string(index=False))
    print({m: (len(p), len(k)) for m, (p, k) in pools.items()}, "max_error", five["max_error"], "recoded", five["n_recoded"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
