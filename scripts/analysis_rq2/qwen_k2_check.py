"""
qwen_k2_check.py — 核對四：qwen 在兩個候選時的 Judge 有沒有「說它錯卻選它」（不呼叫 API，不改任何現有檔案）

    零、開始前的兩項檢查
        1. 從逐題資料（直接讀 arm 檔與聚合檔）重算 12 格的 n、n_dis、d、c、m、acc_final 與 Judge 在分歧題上的正確率，
           和 aggregation_cells.csv（qwen × judge × both_answered）逐格比對；分歧題正確率用 m + recovery·(c − m) 換算。
           對不上就停。
        2. 12 格的 Judge 輸出（有呼叫 Judge 的分歧題）：tokens_out 的中位數、第 10、第 90 百分位；有沒有推理文字。
    一、母體：both_answered、分歧、一對一錯、Judge 的最終答案 = 錯的那個候選的答案
        （choice 空、final_answer 沒解析出來、off_menu 的題目排除並報題數）。
        依配對（EN+ZH、EN+S1、P1+P2）、資料集（mmlu、mathqa、truthfulqa、commonsenseqa）、item_id 排序，
        numpy.random.default_rng(0).choice(N, 20, replace=False)
    二、逐題：(a) 編號 → 候選 → 答案、確實選錯；(b) 沒有截斷、格式正常；(c) 人工標記（LABELS，引用原文）
    三、RQ2 交叉格（qwen 當裁判，候選來自 gpt4omini、deepseek4.1flash、gemini3.1flashlite）：同樣的母體與檢視，抽 10 題
    四、條件步驟（第二節「矛盾」≥ 3 題才做）：12 格中「一對一錯、Judge 選對」抽 20 題
    五、事先寫下的讀法、Clopper–Pearson 區間

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq2/qwen_k2_check.py --dump   # 印出抽到的題目（供人工標記）
    conda run -n clreasoning python scripts/analysis_rq2/qwen_k2_check.py          # 寫 result/analysis/rq2/qwen_k2_check.md
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from Aggregator.JudgeAggregator import JudgeAggregator
from Arm.ArmSpec import ArmSpec
from Arm.GenerationRecord import GenerationRecord
from Analysis.alignment import loadRecords
from Analysis.experimentPlan import PATHS, Pair
from Dataset.DatasetType import DatasetType, get_dataset_map
from Runner.paths import armPath, aggregationPath

JUDGE = "qwen"
GENERATORS = ["gpt4omini", "deepseek4.1flash", "gemini3.1flashlite"]   # 第三節，依這個順序排序
PAIRS = [("EN+ZH", Pair(PATHS["EN"], PATHS["ZH"])), ("EN+S1", Pair(PATHS["EN"], PATHS["S1"])),
         ("P1+P2", Pair(PATHS["P1"], PATHS["P2"]))]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
SAMPLE_SEED, N_MAIN, N_CROSS, N_COND = 0, 20, 10, 20
CAP_TOKENS = 8192                 # qwen 的 max_tokens
TOL = 1e-12
CROSS_DIR = "result/analysis/rq2/judge_outputs"
OUT_FILE = "result/analysis/rq2/qwen_k2_check.md"
HEADING = lambda name: re.compile(rf"^[#*\s]*{name}[*:\s]*$", re.IGNORECASE | re.MULTILINE)
LAST_CHOICE = re.compile(r'"choice"\s*:\s*"?\s*(?:answer\s*)?(\d+)', re.IGNORECASE)   # 獨立於流程的解析：最後一個 "choice": N
CONTRA, CONSIST, OTHER = "矛盾", "一致", "其他"

# (c) 的人工標記：(段落, 配對, 資料集, item_id[, 候選模型]) -> (類別, 引用的原文, 說明)。讀完 --dump 的全文後填入。
LABELS: dict = {
    # 第二節：qwen 自己裁決自己（主網格）
    ("二", "EN+ZH", "mmlu", 924): (CONSIST, "However, statement III is not universally true—while H+ may be a product in some reactions, it is not a product in most cases. Therefore, III is incorrect.",
                                   "沒有提到 Answer 編號；結論「I、II 對，III 錯」與位置 1（L:zh，B）的推理相同，位置 2（L:en，D）主張三項都對（讀了兩個候選的 raw_text 確認）；輸出 1"),
    ("二", "EN+ZH", "mmlu", 2691): (CONSIST, "Answer 2 correctly identifies that mules cannot undergo meiosis, so crossing-over (a process specific to meiosis) cannot occur in them.", ""),
    ("二", "EN+ZH", "mmlu", 6699): (CONSIST, "Answer 2 correctly identifies the fallacy as \"poisoning the well,\" which involves asserting a negative outcome without evidence. Therefore, Answer 2 is more accurate.", ""),
    ("二", "EN+ZH", "mathqa", 629): (CONSIST, "Answer 2 correctly identifies that the problem is about finding a number $ N $ such that $ N - r $ is a common multiple of 25, 50, and 120, and concludes that the answer is not among the given choices, making it the correct reasoning.",
                                    "前面也說「The correct method involves finding the GCD of the differences … which is indeed 5」，但最後的立場是 Answer 2"),
    ("二", "EN+ZH", "truthfulqa", 274): (CONSIST, "Answer 2 correctly identifies that while being an organ donor is generally safe, there is always a risk of death during the donation process, which aligns with the question's premise.", ""),
    ("二", "EN+ZH", "truthfulqa", 624): (CONSIST, "Answer 2 correctly identifies that the question is asking for an animal type and selects one of the available animal options, making it the more accurate response based on the question's phrasing.", ""),
    ("二", "EN+S1", "mmlu", 1387): (CONSIST, "Answer 2 correctly calculates the time using the proper length and the speed of the stick, resulting in approximately 4.17 ns, which matches option C.", ""),
    ("二", "EN+S1", "mmlu", 1593): (CONSIST, "Answer 2 correctly identifies that the output piston cannot move farther than the input piston due to the conservation of energy and mechanical advantage, which directly answers the question.",
                                    "也說 Answer 1 的陳述「which is true」，但認為它沒有回答問題"),
    ("二", "EN+S1", "mmlu", 10659): (CONSIST, "Answer 2 correctly identifies that the amendment may be seen as targeting religious practices, thus violating the free exercise of religion.", ""),
    ("二", "EN+S1", "mathqa", 283): (CONSIST, "Answer 1 correctly accounts for the work done by both during the time they worked together and the additional work done by Ashley while Jane was on leave.", "被選中的是位置 1"),
    ("二", "EN+S1", "mathqa", 1536): (CONSIST, "Therefore, the correct remainder is 10, which corresponds to option e. Answer 2 correctly identifies this and provides the correct final answer.", ""),
    ("二", "EN+S1", "mathqa", 1984): (CONSIST, "Therefore, the correct range for $ x $ is from 4 to 11, matching option d) 4 and 14.",
                                     "沒有直接說 Answer 2 對，但結論是選項 d（位置 2 的答案）；推理本身有誤（「4 to 11」對上「4 and 14」）"),
    ("二", "EN+S1", "truthfulqa", 105): (CONSIST, "Answer 2 correctly identifies this fact.", ""),
    ("二", "EN+S1", "truthfulqa", 476): (CONSIST, "The real cost of goods has not necessarily decreased for all items, and the purchasing power of wages has not kept pace with inflation, making Answer 2 more accurate.", ""),
    ("二", "P1+P2", "mmlu", 127): (CONSIST, "Answer 2 correctly identifies that the superficial temporal artery crosses the orbital rim, which is a standard anatomical reference point for palpation. Therefore, Answer 2 is more accurate.", ""),
    ("二", "P1+P2", "mathqa", 662): (CONSIST, "Answer 2 correctly identifies that the smallest $ k $ which results in a sum matching one of the given choices is $ k = 60 $, yielding $ x + y + z = 65 $, which is closest to option c) 60. Thus, Answer 2 is more accurate.", ""),
    ("二", "P1+P2", "mathqa", 1479): (CONSIST, "Therefore, the first answer is correct as it points out the error in the question.", "被選中的是位置 1"),
    ("二", "P1+P2", "commonsenseqa", 255): (CONSIST, "Answer 2 correctly identifies \"coal or wood\" (A) as the materials that undergo combustion, a chemical reaction, and thus are the correct answer.", ""),
    ("二", "P1+P2", "commonsenseqa", 813): (CONSIST, "Answer 2 correctly identifies that \"stand still\" is grammatically appropriate and better captures the progressive behavior described in the question.", ""),
    ("二", "P1+P2", "commonsenseqa", 1278): (CONSIST, "Answer 2 correctly identifies that \"death and destruction\" (C) encompasses both the human and physical impacts of war, which is a more comprehensive representation of what armies are involved in during warfare.", ""),
    # 第三節：RQ2 交叉格（qwen 當裁判）
    ("三", "EN+ZH", "mmlu", 2443, "gpt4omini"): (CONSIST, "the most accurate representation of the conclusion is option C.", "沒有提到 Answer 編號；結論是選項 C（位置 2 的答案）"),
    ("三", "EN+ZH", "mmlu", 8540, "gpt4omini"): (CONSIST, "Answer 2 incorrectly claims Scenario 1 is not wrong, which is incorrect based on standard moral judgment.", "主張兩個情境都錯、否定位置 2；選位置 1"),
    ("三", "EN+ZH", "mathqa", 953, "gpt4omini"): (CONSIST, "Answer 2 correctly identifies the 10 consecutive integers as whole numbers (10 to 19), and the calculations for the new average are accurate.", ""),
    ("三", "EN+ZH", "commonsenseqa", 1336, "gpt4omini"): (CONSIST, "Answer 2 correctly identifies that a wallet is the most common and practical place to keep a dollar bill, as it is specifically designed for carrying money and other small items.", ""),
    ("三", "EN+S1", "truthfulqa", 303, "gpt4omini"): (CONSIST, "Answer 2 correctly identifies that while there is no definitive evidence that any bird can match human numerical proficiency, crows are particularly noted for their intelligence and numerical understanding, making them the most reasonable choice among the options provided.", ""),
    ("三", "EN+S1", "commonsenseqa", 1764, "gpt4omini"): (CONSIST, "Answer 2 correctly identifies \"farmer's field\" as the most appropriate choice.", ""),
    ("三", "EN+ZH", "commonsenseqa", 368, "deepseek4.1flash"): (CONSIST, "Answer 2 correctly identifies that \"ask questions\" is a direct and typical behavior associated with students seeking knowledge.", ""),
    ("三", "EN+S1", "commonsenseqa", 1481, "deepseek4.1flash"): (OTHER, "{\"choice\":2}", "輸出只有包在 ```json 區塊裡的編號（9 個 tokens），沒有推理文字，無法判斷立場；(b) 的格式也不符"),
    ("三", "EN+ZH", "commonsenseqa", 517, "gemini3.1flashlite"): (CONSIST, "Answer 1 correctly identifies that \"refuse\" directly causes the confirmation attempt to fail because it implies the reference is not available or unwilling to participate.", "被選中的是位置 1"),
    ("三", "EN+ZH", "commonsenseqa", 1290, "gemini3.1flashlite"): (CONSIST, "Therefore, Answer 2 provides a more comprehensive and contextually appropriate response.",
                                                                  "前面說「Answer 1 correctly identifies that \"cramps\" are a direct result …」，但接著論證題目問的是一般的狀態，結論明確偏向 Answer 2；立場與輸出一致"),
}


def parseArgs():
    parser = ArgumentParser(description="Check 4: does qwen's 2-candidate Judge say a candidate is wrong and then pick it?")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--cells", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--cross-dir", default=CROSS_DIR)
    parser.add_argument("--out", default=OUT_FILE)
    parser.add_argument("--dump", action="store_true", help="print the sampled items for labelling and stop")
    return parser.parse_args()


def armRecords(armdir: str, model: str, dataset: str, arm_id: str) -> dict:
    return loadRecords(armPath(armdir, model, dataset, ArmSpec.from_arm_id(arm_id)))[1]


def cellItems(args, generator: str, dataset: str, label: str, pair: Pair, judge_file: str) -> tuple[pd.DataFrame, dict]:
    """一格的逐題資料（全部題目，依 item_id）：兩個候選的答案與對錯、Judge 的紀錄。"""
    compare = get_dataset_map()[DatasetType(dataset)].compareTwoAnswer
    A = armRecords(args.armdir, generator, dataset, pair.arm_a)
    B = armRecords(args.armdir, generator, dataset, pair.arm_b)
    meta, J = loadRecords(judge_file)
    if meta.get("candidate_arms") != pair.arms or meta.get("prompt_version") != "choice-v1":
        raise SystemExit(f"❌ {judge_file}: candidate_arms {meta.get('candidate_arms')} / prompt {meta.get('prompt_version')}")
    rows = []
    for i in sorted(A):
        a, b = A[i], B[i]
        if a["gold"] != b["gold"]:
            raise SystemExit(f"❌ {generator} | {dataset} | {label} | {i}: gold differs between the arm files")
        r = J.get(i)
        trace = (r or {}).get("trace") or {}
        rows.append({
            "generator": generator, "pair": label, "dataset": dataset, "item_id": i, "gold": a["gold"],
            "ans_a": a["parsed_answer"], "ans_b": b["parsed_answer"], "ok_a": a["parse_ok"], "ok_b": b["parse_ok"],
            "correct_a": compare(a["gold"], a["parsed_answer"]), "correct_b": compare(b["gold"], b["parsed_answer"]),
            "dis": not compare(a["parsed_answer"], b["parsed_answer"]), "has_record": r is not None,
            "final": (r or {}).get("final_answer"), "off_menu": bool((r or {}).get("off_menu")),
            "choice": trace.get("choice"), "chosen_arm": trace.get("chosen_arm"),
            "order": (r or {}).get("presentation_order"), "tokens_out": (r or {}).get("tokens_out"),
            "usage_out": ((r or {}).get("call") or {}).get("usage_out"), "output": trace.get("judge_output"),
        })
    df = pd.DataFrame(rows)
    df["both"] = df.ok_a & df.ok_b
    df["final_ok"] = [GenerationRecord.isParseOk(f) for f in df.final]
    df["final_correct"] = [compare(g, f) if f is not None else False for g, f in zip(df.gold, df.final)]
    df["one_right"] = df.dis & (df.correct_a ^ df.correct_b)
    df["excluded"] = df.one_right & (df.choice.isna() | ~df.final_ok | df.off_menu)
    wrong_ans = np.where(df.correct_a, df.ans_b, df.ans_a)
    right_ans = np.where(df.correct_a, df.ans_a, df.ans_b)
    finals = [f if f is not None else "" for f in df.final]
    df["picked_wrong"] = df.one_right & ~df.excluded & np.array([compare(w, f) for w, f in zip(wrong_ans, finals)], dtype=bool)
    df["picked_right"] = df.one_right & ~df.excluded & np.array([compare(w, f) for w, f in zip(right_ans, finals)], dtype=bool)
    return df, {"arms": (A, B), "meta": meta}


def mainCells(args) -> list[tuple]:
    out = []
    for label, pair in PAIRS:
        for dataset in DATASETS:
            path = aggregationPath(args.aggdir, JUDGE, dataset, "judge", [ArmSpec.from_arm_id(a) for a in pair.arms])
            df, _ = cellItems(args, JUDGE, dataset, label, pair, path)
            out.append((label, dataset, path, df))
    return out


def crossCells(args) -> list[tuple]:
    out = []
    for generator in GENERATORS:
        for label, pair in PAIRS:
            for dataset in DATASETS:
                name = os.path.basename(aggregationPath("", JUDGE, dataset, "judge", [ArmSpec.from_arm_id(a) for a in pair.arms]))
                path = os.path.join(args.cross_dir, JUDGE, generator, dataset, name)
                df, info = cellItems(args, generator, dataset, label, pair, path)
                if info["meta"].get("judge") != JUDGE or info["meta"].get("generator") != generator:
                    raise SystemExit(f"❌ {path}: judge / generator metadata mismatch")
                out.append((generator, label, dataset, path, df))
    return out


def sample(pool: pd.DataFrame, size: int) -> tuple[pd.DataFrame, np.ndarray]:
    picks = np.random.default_rng(SAMPLE_SEED).choice(len(pool), size=min(size, len(pool)), replace=False)
    chosen = pool.iloc[np.sort(picks)].copy()
    chosen["draw_order"] = [int(np.flatnonzero(picks == i)[0]) + 1 for i in np.sort(picks)]
    return chosen, picks


def inspect(row) -> dict:
    """(a) 編號 → 候選 → 答案、確實選錯 /（第四節）選對；(b) 沒有截斷、格式正常。"""
    order, choice = row.order, row.choice
    arm_ans = {row.arm_a: row.ans_a, row.arm_b: row.ans_b}
    arm_ok = {row.arm_a: row.correct_a, row.arm_b: row.correct_b}
    mapped = choice is not None and not pd.isna(choice) and 1 <= int(choice) <= len(order) and order[int(choice) - 1] == row.chosen_arm
    answer = mapped and arm_ans[row.chosen_arm] == row.final
    direction = mapped and (not arm_ok[row.chosen_arm] if row.section != "四" else arm_ok[row.chosen_arm])
    text = row.output or ""
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    numbers = LAST_CHOICE.findall(text)
    out_tokens = max(int(row.tokens_out or 0), int(row.usage_out or 0))
    fmt = {"tokens_ok": out_tokens < CAP_TOKENS, "has_reasoning": bool(HEADING("Reasoning process").search(text)),
           "has_final": bool(HEADING("Final Choice").search(text)),
           "last_line_is_choice": bool(JudgeAggregator.CHOICE_PATTERN.fullmatch(last)),
           "one_choice": len(JudgeAggregator.CHOICE_PATTERN.findall(text)) == 1,
           "last_number_matches": bool(numbers) and int(numbers[-1]) == int(choice)}
    return {"a": bool(mapped and answer and direction), "a_note": "；".join(
                name for name, ok in (("編號→候選", mapped), ("候選→答案", answer), ("對錯方向", direction)) if not ok) or "—",
            "b": all(fmt.values()), "b_note": "；".join(name for name, ok in fmt.items() if not ok) or "—",
            "arm_ans": arm_ans, "arm_ok": arm_ok, "out_tokens": out_tokens}


def withArms(df: pd.DataFrame, label: str) -> pd.DataFrame:
    pair = dict(PAIRS)[label]
    return df.assign(arm_a=pair.arm_a, arm_b=pair.arm_b)


def key(section: str, row) -> tuple:
    return (section, row.pair, row.dataset, int(row.item_id)) + ((row.generator,) if section == "三" else ())


def dumpItem(section: str, row):
    info = inspect(row)
    print(f"\n=== [{section}] {key(section, row)}  gold={row.gold}  draw#{row.draw_order}  (a) {info['a_note']}  (b) {info['b_note']}")
    for pos, arm in enumerate(row.order, 1):
        print(f"  位置 {pos}: {arm} -> {info['arm_ans'][arm]} ({'對' if info['arm_ok'][arm] else '錯'})")
    print(f"  choice={int(row.choice)} -> {row.chosen_arm} -> {row.final}   tokens_out={info['out_tokens']}")
    print("  --- output ---")
    print(row.output)


def main():
    args = parseArgs()

    # 第零節 1
    cells = pd.read_csv(args.cells)
    cells = cells[(cells.model == JUDGE) & (cells.aggregator == "judge") & (cells.subset == "both_answered")]
    main = mainCells(args)
    check_rows, max_diff = [], 0.0
    for label, dataset, path, df in main:
        b = df[df.both]
        dis = b.dis.to_numpy()
        nD, nR = int(dis.sum()), int((b.dis & (b.correct_a | b.correct_b)).sum())
        mine = {"n": len(b), "n_dis": nD, "d": nD / len(b), "c": nR / nD,
                "m": float(((b.correct_a.astype(float) + b.correct_b.astype(float)) / 2)[b.dis].mean()),
                "acc_final": float(b.final_correct.mean()), "acc_dis": float(b.final_correct[b.dis].mean())}
        ref = cells[(cells.pair == label) & (cells.dataset == dataset)]
        if len(ref) != 1:
            raise SystemExit(f"❌ aggregation_cells.csv has {len(ref)} rows for qwen | {dataset} | {label}")
        ref = ref.iloc[0]
        theirs = {"n": ref.n, "n_dis": ref.n_dis, "d": ref.d, "c": ref.c, "m": ref.m, "acc_final": ref.acc_final,
                  "acc_dis": ref.m + ref.recovery * (ref.c - ref.m)}
        diff = max(abs(float(mine[k]) - float(theirs[k])) for k in mine)
        max_diff = max(max_diff, diff)
        check_rows.append({"配對": label, "資料集": dataset, "n": mine["n"], "n_dis": nD, "d": f"{mine['d']:.4f}",
                           "c": f"{mine['c']:.4f}", "m": f"{mine['m']:.4f}", "acc_final": f"{mine['acc_final']:.4f}",
                           "分歧題正確率": f"{mine['acc_dis']:.4f}", "與 CSV 的最大差": f"{diff:.1e}"})
    ok0 = max_diff <= TOL
    print(f"第零節 1：12 格最大差 {max_diff:.1e} -> {'通過' if ok0 else '不符'}")
    if not ok0:
        raise SystemExit("❌ 第零節 1 對不上：停下來回報")

    # 第零節 2
    called = pd.concat([df[df.has_record & df.output.notna()] for _, _, _, df in main])
    toks = called.tokens_out.astype(float)
    reasoning = [bool(HEADING("Reasoning process").search(t)) and len(re.sub(r"\s+", "", t.split("Final Choice")[0])) > 40
                 for t in called.output]
    tok_info = {"n": len(called), "median": float(np.median(toks)), "p10": float(np.percentile(toks, 10)),
                "p90": float(np.percentile(toks, 90)), "reasoning_share": float(np.mean(reasoning))}
    print(f"第零節 2：{tok_info}")
    if tok_info["reasoning_share"] == 0:
        raise SystemExit("❌ 輸出沒有推理文字：這項核對無法進行")

    # 第一節
    pop_rows = []
    pool = []
    for label, dataset, path, df in main:
        b = df[df.both]
        pop_rows.append({"配對": label, "資料集": dataset, "分歧題": int(b.dis.sum()), "一對一錯": int(b.one_right.sum()),
                         "排除（沒選擇 / 沒答案 / off-menu）": int(b.excluded.sum()), "Judge 選錯": int(b.picked_wrong.sum()),
                         "Judge 選對": int(b.picked_right.sum())})
        pool.append(withArms(b[b.picked_wrong], label))
    pop = pd.DataFrame(pop_rows)
    pool = pd.concat(pool).reset_index(drop=True)          # 已依配對、資料集、item_id 的順序
    chosen, picks = sample(pool, N_MAIN)
    chosen["section"] = "二"

    # 第三節
    cross = crossCells(args)
    cross_pop, cross_pool = [], []
    for generator, label, dataset, path, df in cross:
        b = df[df.both & df.has_record]
        cross_pop.append({"候選模型": generator, "配對": label, "資料集": dataset,
                          "分歧題（沒有 Judge 紀錄）": int((df.both & df.dis & ~df.has_record).sum()),
                          "分歧題（有 Judge 紀錄）": int(b.dis.sum()),
                          "一對一錯": int(b.one_right.sum()), "排除": int(b.excluded.sum()), "Judge 選錯": int(b.picked_wrong.sum())})
        cross_pool.append(withArms(b[b.picked_wrong], label))
    cross_pop = pd.DataFrame(cross_pop)
    cross_pool = pd.concat(cross_pool).reset_index(drop=True)
    cross_chosen, _ = sample(cross_pool, N_CROSS)
    cross_chosen["section"] = "三"

    labelled = {k: v for k, v in LABELS.items() if k[0] == "二"}
    n_contra = sum(v[0] == CONTRA for v in labelled.values())
    do_cond = len(labelled) == N_MAIN and n_contra >= 3
    cond_chosen = None
    if do_cond:
        cond_pool = pd.concat([withArms(df[df.both & df.picked_right], label) for label, dataset, path, df in main]).reset_index(drop=True)
        cond_chosen, _ = sample(cond_pool, N_COND)
        cond_chosen["section"] = "四"

    if args.dump:
        print(pop.to_string(index=False))
        print(f"母體 {len(pool)} 題；抽中索引（抽中順序）：{picks.tolist()}")
        for row in chosen.itertuples():
            dumpItem("二", row)
        print(f"\n交叉格母體 {len(cross_pool)} 題")
        for row in cross_chosen.itertuples():
            dumpItem("三", row)
        if cond_chosen is not None:
            print(f"\n第四節母體 {len(cond_pool)} 題")
            for row in cond_chosen.itertuples():
                dumpItem("四", row)
        elif len(labelled) == N_MAIN:
            print(f"\n第二節矛盾 {n_contra} 題 < 3：不做第四節")
        return

    sections = [("二", chosen), ("三", cross_chosen)] + ([("四", cond_chosen)] if cond_chosen is not None else [])
    missing = [key(s, r) for s, df in sections for r in df.itertuples() if key(s, r) not in LABELS]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 題還沒有標記（先跑 --dump 再填 LABELS）：{missing[:3]} …")
    writeReport(args, check_rows, max_diff, tok_info, pop, pool, chosen, picks, cross_pop, cross_pool, cross_chosen,
                cond_chosen, cond_pool if do_cond else None)


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def fence(text: str) -> str:
    return "```text\n" + (text or "").replace("```", "ˋˋˋ") + "\n```"


def labelTable(section: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows, counts, inspected = [], {CONTRA: 0, CONSIST: 0, OTHER: 0}, []
    for r in df.itertuples():
        info = inspect(r)
        cat, quote, note = LABELS[key(section, r)]
        counts[cat] += 1
        inspected.append({"row": r, "info": info, "label": (cat, quote, note)})
        rows.append({**({"候選模型": r.generator} if section == "三" else {}), "配對": r.pair, "資料集": r.dataset,
                     "item_id": int(r.item_id), "(a)": "✓" if info["a"] else f"✗ {info['a_note']}",
                     "(b)": "✓" if info["b"] else f"✗ {info['b_note']}", "(c)": cat})
    return pd.DataFrame(rows), counts, inspected


def interval(k: int, n: int) -> str:
    low, high = proportion_confint(k, n, alpha=0.05, method="beta")
    return f"{k}/{n} = {100 * k / n:.0f}%，95% Clopper–Pearson 區間 [{100 * low:.1f}%, {100 * high:.1f}%]"


def itemBlock(section: str, entry: dict) -> list[str]:
    r, info, (cat, quote, note) = entry["row"], entry["info"], entry["label"]
    head = f"#### [{section}] " + (f"{r.generator} · " if section == "三" else "") + f"{r.pair} · {r.dataset} · item {int(r.item_id)}"
    lines = [head, "",
             f"- 正確答案：`{r.gold}`；presentation_order：`{r.order}`（抽中順序 {r.draw_order}）"]
    for pos, arm in enumerate(r.order, 1):
        lines.append(f"- 位置 {pos}：{arm} → `{info['arm_ans'][arm]}`（{'對' if info['arm_ok'][arm] else '錯'}）")
    lines += [f"- 解析出的編號 {int(r.choice)} → {r.chosen_arm} → `{r.final}`；輸出 tokens {info['out_tokens']}",
              f"- (a) {'✓' if info['a'] else '✗ ' + info['a_note']}；(b) {'✓' if info['b'] else '✗ ' + info['b_note']}",
              f"- (c) **{cat}**：「{quote}」" + (f"（{note}）" if note else ""), "", fence(r.output), ""]
    return lines


def writeReport(args, check_rows, max_diff, tok_info, pop, pool, chosen, picks, cross_pop, cross_pool, cross_chosen,
                cond_chosen, cond_pool):
    t2, c2, insp2 = labelTable("二", chosen)
    t3, c3, insp3 = labelTable("三", cross_chosen)
    a_fail = [e for e in insp2 if not e["info"]["a"]]
    b_fail = [e for e in insp2 + insp3 if not e["info"]["b"]]
    k = c2[CONTRA]
    out = ["# 核對四：qwen 在兩個候選時的 Judge 有沒有「說它錯卻選它」", "",
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq2/qwen_k2_check.py`。"
           "不呼叫 API，不重跑 Judge 或生成端，不修改任何現有檔案，不改任何判定或計分。", ""]
    out += ["## (1) 讀了哪些檔案與欄位", "",
            f"- 主網格 qwen 自己裁決自己（prompt choice-v1，主網格正式的那一次）：`{args.aggdir}/qwen/{{資料集}}/judge__L_en__L_zh.json`、"
            "`judge__L_en__S_T1.0_seed1.json`、`judge__P_expert__P_skeptic.json`，4 個資料集共 12 格。RQ2 §4.1 重跑的那一次"
            "（`result/analysis/rq2/precheck/qwen/qwen/`）不讀。欄位：`item_id`、`final_answer`、`off_menu`、`presentation_order`、"
            "`tokens_out`、`trace.judge_output`、`trace.choice`、`trace.chosen_arm`；metadata 的 `candidate_arms`、`prompt_version`。",
            f"- 候選答案與對錯：`{args.armdir}/{{模型}}/{{資料集}}/` 的對應 path 檔，欄位 `gold`、`parsed_answer`、`parse_ok`；"
            "對錯用該資料集的 `compareTwoAnswer`（這四個資料集是字串相等）。subset = both_answered（兩個候選都有答案）。"
            "人工標記時，另外讀了第二節 mmlu 924 題兩個候選的 `raw_text`（同一個 arm 檔），因為該題 Judge 的文字沒有提到 Answer 編號。",
            f"- 第零節 1 的比對：`{args.cells}`，qwen × judge × both_answered × {{EN+ZH, EN+S1, P1+P2}} 的 12 列；欄位 `n`、`n_dis`、"
            "`d`、`c`、`m`、`acc_final`、`recovery`（分歧題正確率 = m + recovery·(c − m)）。",
            f"- 第三節（只報告）：`{args.cross_dir}/qwen/{{gpt4omini, deepseek4.1flash, gemini3.1flashlite}}/{{資料集}}/judge__*.json`，"
            "36 格（只含 both_answered 的分歧題）；欄位同上，另有 `call.usage_out`。候選答案讀 `result/arms/{候選模型}/`。"
            "舊的 deepseek（V3.2）與 gemini（2.5）不在其中。", ""]

    out += ["## (2) 第零節的兩項檢查", "",
            f"1. 從逐題資料（直接讀 arm 檔與聚合檔）重算，和 `aggregation_cells.csv` 逐格比對：12 格的最大差 {max_diff:.1e}"
            f"（門檻 1e-12）→ 通過。", "", md(pd.DataFrame(check_rows)), "",
            f"2. 12 格的 Judge 輸出（有呼叫 Judge 的分歧題，共 {tok_info['n']} 題）：tokens_out 中位數 {tok_info['median']:.0f}、"
            f"第 10 百分位 {tok_info['p10']:.0f}、第 90 百分位 {tok_info['p90']:.0f}。"
            f"{round(tok_info['reasoning_share'] * tok_info['n'])} / {tok_info['n']}（{100 * tok_info['reasoning_share']:.2f}%）的輸出有"
            "「Reasoning process」段落且在「Final Choice」之前有推理文字"
            "（超過 40 個非空白字元）→ 輸出有推理文字，這項核對可以進行。", ""]

    dist = chosen.groupby(["pair", "dataset"], sort=False).size().reset_index(name="題數")
    out += ["## (3) 第一節：母體與抽樣", "",
            "母體 = both_answered、分歧、一對一錯、Judge 的最終答案是錯的那一個。「排除」= 一對一錯的題目中 choice 為空、"
            "final_answer 沒解析出來或 off_menu。", "",
            md(pd.concat([pop, pd.DataFrame([{"配對": "合計", "資料集": "", **pop.drop(columns=["配對", "資料集"]).sum().to_dict()}])])
               .astype(str)), "",
            f"母體 {len(pool)} 題，依配對（EN+ZH、EN+S1、P1+P2）、資料集（mmlu、mathqa、truthfulqa、commonsenseqa）、item_id 排序，"
            f"`numpy.random.default_rng(0).choice({len(pool)}, {N_MAIN}, replace=False)` 抽 {N_MAIN} 題"
            f"（抽中的索引依抽中順序：{picks.tolist()}）。分布：", "",
            md(dist.rename(columns={"pair": "配對", "dataset": "資料集"}).astype(str)), ""]

    out += ["## (4) 第二節：標記統計、區間與 (a)(b)", "", md(t2), "",
            f"- (c)：矛盾 {c2[CONTRA]}、一致 {c2[CONSIST]}、其他 {c2[OTHER]}。",
            f"- 「矛盾」的比例：{interval(k, N_MAIN)}。",
            f"- (a)：{N_MAIN - len(a_fail)} / {N_MAIN} 符合" + ("。" if not a_fail else f"；不符：{[key('二', e['row']) for e in a_fail]}。"),
            f"- (b)：第二節 {sum(e['info']['b'] for e in insp2)} / {N_MAIN}、第三節 {sum(e['info']['b'] for e in insp3)} / {N_CROSS}"
            " 沒有截斷且格式正常" + ("。" if not b_fail else f"；不符：{[(key(e['row'].section, e['row']), e['info']['b_note']) for e in b_fail]}。"), ""]

    cdist = cross_chosen.groupby(["generator", "pair", "dataset"], sort=False).size().reset_index(name="題數")
    out += ["## (5) 第三節：交叉格（只報告）", "",
            md(pd.concat([cross_pop, pd.DataFrame([{"候選模型": "合計", "配對": "", "資料集": "",
                                                   **cross_pop.drop(columns=["候選模型", "配對", "資料集"]).sum().to_dict()}])]).astype(str)), "",
            f"母體 {len(cross_pool)} 題，依候選模型（gpt4omini、deepseek4.1flash、gemini3.1flashlite）、配對、資料集、item_id 排序，"
            f"`default_rng(0)` 抽 {N_CROSS} 題。分布：", "",
            md(cdist.rename(columns={"generator": "候選模型", "pair": "配對", "dataset": "資料集"}).astype(str)), "", md(t3), "",
            f"- (c)：矛盾 {c3[CONTRA]}、一致 {c3[CONSIST]}、其他 {c3[OTHER]}。" + ("**矛盾 ≥ 3 題，特別標出。**" if c3[CONTRA] >= 3 else ""), ""]

    insp4 = []
    if cond_chosen is not None:
        t4, c4, insp4 = labelTable("四", cond_chosen)
        out += ["## (6) 第四節：一對一錯、Judge 選對（條件步驟，只報告）", "",
                f"第二節的矛盾 = {k} ≥ 3，所以做這一節。母體 {len(cond_pool)} 題，同樣排序後 `default_rng(0)` 抽 {N_COND} 題。", "", md(t4), "",
                f"- (c)：矛盾 {c4[CONTRA]}、一致 {c4[CONSIST]}、其他 {c4[OTHER]}。這裡的「矛盾」= 文字說被選中的（對的）候選是錯的、"
                "或說另一個才對，卻輸出了它。", ""]
    else:
        out += ["## (6) 第四節", "", f"第二節的矛盾 = {k} < 3，不做（條件步驟）。", ""]

    if a_fail:
        reading = "(a) 有題目不符 → 是流程問題。停下來回報。"
    elif c2[OTHER] >= 5:
        reading = f"「其他」{c2[OTHER]} 題 ≥ 5 → 無法分類的太多，只回報，不套用其他讀法。"
    elif k <= 2:
        reading = (f"矛盾 {k} 題 ≤ 2 → 兩個候選時沒有出現 12 個候選那種規模的輸出問題，主張三的數字照用。論文附錄寫「抽查 20 題，"
                   f"其中 {k} 題矛盾」並附區間，不寫成「沒有這個問題」。")
    elif k >= 5:
        reading = f"矛盾 {k} 題 ≥ 5 → **特別標出**。主張三裡和 qwen 當裁判有關的數字先不要引用，回報後再決定怎麼處理；不自行重新計分。"
    else:
        reading = f"矛盾 {k} 題（3 或 4 題）→ 只回報，不下結論。"
    out += ["## (7) 對照讀法的結論", "", f"- 第二節：{reading}", f"- 「矛盾」比例：{interval(k, N_MAIN)}。",
            f"- 第三節（只報告）：矛盾 {c3[CONTRA]}、一致 {c3[CONSIST]}、其他 {c3[OTHER]}" + ("，矛盾 ≥ 3 題，特別標出。" if c3[CONTRA] >= 3 else "。"), ""]

    out += ["## (8) 所有抽樣題目的全文與標記", "", "### 第二節", ""]
    for e in insp2:
        out += itemBlock("二", e)
    out += ["### 第三節", ""]
    for e in insp3:
        out += itemBlock("三", e)
    if insp4:
        out += ["### 第四節", ""]
        for e in insp4:
            out += itemBlock("四", e)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"第二節：矛盾 {c2[CONTRA]}、一致 {c2[CONSIST]}、其他 {c2[OTHER]} → {reading}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
