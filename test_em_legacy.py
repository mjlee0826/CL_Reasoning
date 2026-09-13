"""
test_em_legacy.py — 計算舊版 (2025-11 世代) 結果檔的 Exact Match

為什麼需要獨立腳本:
  舊檔與現行 pipeline 不相容, 無法走 File / FileFactory / TestEM 那條路
    1. metadata 是扁平字串  {"Model": "Deepseek", "Dataset": "MMLU", "Strategy": "Challenge"}
       -> DatasetConfig.from_dict("MMLU") 會 AttributeError
    2. records 完全沒有 "id" 欄位
       -> File.records_map 會是空的, TestEM 會靜默回傳 accuracy 0.0
  本腳本直接讀 JSON (不經過 File), 也不會改寫原始檔案; 只 import Test/TestRecoveryBlind.py 的共用計算。

比對邏輯:
  Dataset 家族中只有 MGSM 覆寫 compareTwoAnswer, 而 MGSM 不在這批資料裡,
  因此一律使用基底類別的 str(Answer) == str(MyAnswer)。

除了 EM 之外也一併輸出 Times (辯論回合數) 的統計 — 同語言辯論常常兩個 agent
一開始就同意 (Times=0, 根本沒辯), 比較同語言 vs 跨語言時必須控制這個「觸發率」,
否則會把「跨語言更常觸發辯論」誤讀成「跨語言本身比較好」。

另外計算 c = P(至少一個 agent 答對 | 兩者不一致):
  Times>0 恰好等價於「兩個 agent 初始答案不一致」(Challenge 只在初始不同時才進辯論迴圈),
  而 AnswerRecord1[0] / AnswerRecord2[0] 是兩者的初始答案。
  c 是辯論在分歧子集上的天花板 — 就算裁判完美, 最多也只能到 c。

0A-1 recovery_blind (與 Test/TestRecoveryBlind.py 共用 TestRecoveryBlind.compute):
  一致題 (Times=0) 沒有存 AnswerRecord, 但兩個初答相同且就是 MyAnswer; 分歧題取 AnswerRecord[0]。
  d, c, m 用全部題目 (d = debate_rate, c = c_recoverable);
  錨點 = agent1 / agent2 中在 H1 答對較多者, n_A / n_B / w_A / recovery_blind / recovery / skill
  在 H2 計算, 對 200 次切分取平均 (欄位加 _H2 字尾)。

用法 (會 import numpy 與 Test, 請在 clreasoning 環境執行):
    conda run -n clreasoning python test_em_legacy.py result/tempature1/challenge_EN result/tempature1/challenge_CN
    conda run -n clreasoning python test_em_legacy.py result/tempature1                    # 遞迴, 含 baseline 與各子目錄
    conda run -n clreasoning python test_em_legacy.py result/tempature1/challenge_EN result/tempature1/challenge_CN --csv legacy_em_samelang.csv
"""

import csv
import json
import os
import sys
from argparse import ArgumentParser

from Test.TestRecoveryBlind import TestRecoveryBlind

# 0A-1 recovery_blind 的輸出欄位 (d, c 沿用 debate_rate, c_recoverable)
RB_FIELDS = ["m", "anchor", "anchor_rate", "n_A_H2", "n_B_H2", "w_A_H2",
             "recovery_blind_H2", "recovery_H2", "skill_H2", "reps_undefined", "reps_skill_undefined"]


def parse_name(path):
    """從檔名解析 (model, dataset, strategy)。

    舊檔命名: '{Model}_{Dataset}_{Strategy}.json'
    model / dataset / strategy 內可能有空格與連字號, 但都沒有底線,
    例如 'GPT 4.1 mini_CMB-Exam_Challenge.json' -> 3 段。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return stem, "?", "?"


def condition_of(path, strategy):
    """實驗條件: challenge 子目錄用目錄名 (challenge_EN / challenge_CN / challenge_CNEN),
    直接放在上層的 baseline 檔用檔名裡的 strategy (Only English / Only Chinese)。"""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent.startswith("challenge"):
        return parent
    return strategy


def evaluate(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("檔案為空或不是 [metadata, record...] 格式")

    records = data[1:]
    model, dataset, strategy = parse_name(path)

    total = len(records)
    correct = 0
    empty = 0
    # Times=0 代表兩個 agent 一開始就同意, 沒有真的辯論
    n_no_debate = n_debate = 0
    c_no_debate = c_debate = 0
    has_times = False
    # c = P(至少一個 agent 答對 | 兩者不一致)
    n_disagree = n_recoverable = 0
    # 0A-1 recovery_blind 用的逐題陣列: agent1 / agent2 初答是否正確、最終是否正確、初答是否不同
    init1_ok, init2_ok, final_ok, init_diff = [], [], [], []
    n_missing_inits = 0

    for r in records:
        ans = str(r.get("Answer", ""))
        my = str(r.get("MyAnswer", ""))
        ok = (ans == my)

        if ok:
            correct += 1
        if my.strip() in ("", "None", "null"):
            empty += 1

        times = r.get("Times")
        if times is not None:
            has_times = True
            if times > 0:
                n_debate += 1
                c_debate += ok
                # Times>0 <=> 兩個 agent 初始答案不一致 (Challenge 只在初始不同時才進辯論迴圈)
                # AnswerRecord[0] 是各自的初始答案
                r1 = r.get("AnswerRecord1") or []
                r2 = r.get("AnswerRecord2") or []
                inits = [str(rec[0]) for rec in (r1, r2) if rec]
                if inits:
                    n_disagree += 1
                    if any(x == ans for x in inits):
                        n_recoverable += 1
                if r1 and r2:
                    a1, a2 = str(r1[0]), str(r2[0])
                else:
                    n_missing_inits += 1
                    a1 = a2 = my
            else:
                n_no_debate += 1
                c_no_debate += ok
                # 一致題沒有存 AnswerRecord, 但兩個初答相同, 且就是最終答案
                a1 = a2 = my
            init1_ok.append(a1 == ans)
            init2_ok.append(a2 == ans)
            final_ok.append(ok)
            init_diff.append(a1 != a2)

    rb = {}
    if has_times and n_missing_inits == 0 and len(init1_ok) == total:
        rb = TestRecoveryBlind.compute(init1_ok, init2_ok, final_ok, init_diff,
                                       "agent1", "agent2", 0, 1)
        # 與上面既有的 d、c 必須一致 (Times>0 <=> 初答不同)
        assert rb["D"] == n_debate == n_disagree, f"{path}: 分歧題數不一致"
        if n_disagree:
            assert abs(rb["c"] - n_recoverable / n_disagree) < 1e-12, f"{path}: c 不一致"
    elif has_times:
        print(f"⚠️  {os.path.basename(path)}: {n_missing_inits} 筆分歧題缺 AnswerRecord, 略過 recovery_blind")

    return {
        "file": os.path.basename(path),
        "path": path,
        "model": model,
        "dataset": dataset,
        "strategy": strategy,
        "condition": condition_of(path, strategy),
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "empty_myanswer": empty,
        "has_times": has_times,
        "n_debate": n_debate,
        "n_no_debate": n_no_debate,
        # 觸發率: 實際發生辯論的題目比例
        "debate_rate": n_debate / total if (has_times and total) else None,
        # 分層 EM: 有辯 vs 沒辯
        "acc_debate": c_debate / n_debate if n_debate else None,
        "acc_no_debate": c_no_debate / n_no_debate if n_no_debate else None,
        # c = P(至少一個 agent 答對 | 兩者不一致) — 辯論在分歧子集上的天花板
        "n_disagree": n_disagree,
        "c_recoverable": n_recoverable / n_disagree if n_disagree else None,
        # 回收率: 在 [隨機挑一邊, 完美裁判] 之間的位置。
        # 不一致 => 至多一個 agent 答對, 所以隨機挑一邊的期望正確率是 c/2, 那才是下限。
        #   recovery = (acc - c/2) / (c - c/2) = 2*acc/c - 1
        "recovery": (2 * (c_debate / n_debate) / (n_recoverable / n_disagree) - 1)
                    if (n_debate and n_disagree and n_recoverable) else None,
        # 0A-1 recovery_blind: d, c 即上面的 debate_rate / c_recoverable;
        # *_H2 = 錨點在 H1 決定、在 H2 計算、對 200 次切分取平均
        **{k: rb.get(k) for k in RB_FIELDS},
    }


def collect_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                paths.extend(os.path.join(root, f) for f in files if f.endswith(".json"))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            print(f"[warn] 找不到: {item}")
    return sorted(set(paths))


def fmt_pct(x, width=7):
    return f"{x:>{width}.2%}" if x is not None else " " * (width - 1) + "-"


def print_detail(rows):
    print("\n📋 逐檔明細")
    print(f"{'condition':<15} {'model':<15} {'dataset':<12} {'n':>6} {'EM':>8} "
          f"{'觸發率':>8} {'不一致':>7} {'c':>8} {'EM|有辯':>9} {'回收率':>8}")
    print("-" * 108)
    for r in rows:
        print(f"{r['condition']:<15} {r['model']:<15} {r['dataset']:<12} {r['n']:>6} "
              f"{fmt_pct(r['accuracy'], 8)} {fmt_pct(r['debate_rate'], 8)} "
              f"{r['n_disagree']:>7} {fmt_pct(r['c_recoverable'], 8)} "
              f"{fmt_pct(r['acc_debate'], 9)} {fmt_pct(r['recovery'], 8)}")


def print_pivot(rows, value_key, title, note=""):
    """Row = (model, dataset), Column = condition。"""
    conditions = sorted({r["condition"] for r in rows})
    keys = sorted({(r["model"], r["dataset"]) for r in rows})
    table = {(r["model"], r["dataset"], r["condition"]): r[value_key] for r in rows}

    print(f"\n📊 {title}")
    if note:
        print(f"   {note}")
    head = f"{'model':<16} {'dataset':<12}" + "".join(f"{c:>18}" for c in conditions)
    print(head)
    print("-" * len(head))
    for model, dataset in keys:
        line = f"{model:<16} {dataset:<12}"
        for c in conditions:
            line += fmt_pct(table.get((model, dataset, c)), 18)
        print(line)


def main():
    parser = ArgumentParser(description="計算舊版結果檔的 Exact Match (不改寫原始檔案)")
    parser.add_argument("inputs", nargs="+", help="要掃描的目錄或檔案 (目錄會遞迴)")
    parser.add_argument("--csv", help="把逐檔明細另存成 CSV")
    args = parser.parse_args()

    paths = collect_paths(args.inputs)
    if not paths:
        print("❌ 沒有找到任何 JSON 檔案")
        sys.exit(1)

    print(f"🔍 找到 {len(paths)} 個 JSON 檔案，開始計算 Exact Match...")

    rows = []
    for p in paths:
        try:
            rows.append(evaluate(p))
        except Exception as e:
            print(f"⚠️  略過 {p}: {e}")

    if not rows:
        print("❌ 沒有任何檔案成功解析")
        sys.exit(1)

    rows.sort(key=lambda r: (r["condition"], r["dataset"], r["model"]))

    print_detail(rows)
    print_pivot(rows, "accuracy", "Exact Match 準確率  (row = model × dataset, col = 條件)")
    if any(r["has_times"] for r in rows):
        print_pivot(rows, "debate_rate", "辯論觸發率 (Times > 0 的題目比例)",
                    note="同語言 vs 跨語言比較時必須控制這一項")
        print_pivot(rows, "c_recoverable",
                    "c = P(至少一個 agent 答對 | 兩者不一致)",
                    note="辯論在分歧子集上的天花板：完美的裁判最多能到這個數字")
        print_pivot(rows, "acc_debate", "EM | 有實際辯論的子集",
                    note="⚠️ 各條件辯論的是不同題目子集，跨條件比較會有選擇效應")
        print_pivot(rows, "recovery", "回收率 = (EM|有辯 − c/2) ÷ (c − c/2)",
                    note="0% = 等同隨機挑一邊(c/2)，100% = 完美裁判(c)")

    if args.csv:
        fields = ["condition", "model", "dataset", "strategy", "n", "correct", "accuracy",
                  "empty_myanswer", "n_debate", "n_no_debate", "debate_rate",
                  "n_disagree", "c_recoverable", "acc_debate", "recovery",
                  "acc_no_debate", "file"] + RB_FIELDS
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n💾 明細已匯出至: {args.csv}")


if __name__ == "__main__":
    main()
