"""
check_router_concentration.py — 檢查 router (XLM-R) top-1 預測的集中度

判讀重點:
  - top-1 集中度越高  -> router 越接近「永遠押同一對」，per-question 訊號越少
  - 有效對數 (effective pairs) 接近 1 -> 等同固定 baseline；接近 10 -> 充分分散
  - top1/top2 機率邊際越小 -> argmax 幾乎是在雜訊上做決定
  - 同一題跨 5 個語言版本的選擇若完全一致，代表 router 對輸入內容不敏感

用法:
    python3 scripts/router/check_router_concentration.py result/final_gpt.json
    python3 scripts/router/check_router_concentration.py result/final_*.json
    python3 scripts/router/check_router_concentration.py result/final_gpt.json --by dataset
    python3 scripts/router/check_router_concentration.py result/final_gpt.json --by language
"""

import glob
import json
import math
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions") if isinstance(data, dict) else None
    if not preds:
        raise ValueError(f"{path} 內找不到 predictions")
    return preds


def get_labels(preds):
    """從 predicted_probabilities 的 key 順序取得 label 名稱（與 y 的 index 對應）。"""
    return list(preds[0]["predicted_probabilities"].keys())


def concentration(pairs, n_labels):
    """給一串 best_predicted_pair，算集中度指標。"""
    total = len(pairs)
    counter = Counter(pairs)
    top_pair, top_n = counter.most_common(1)[0]

    shares = [v / total for v in counter.values()]
    entropy = max(0.0, -sum(p * math.log2(p) for p in shares))
    max_entropy = math.log2(n_labels)

    return {
        "total": total,
        "top_pair": top_pair,
        "top_share": top_n / total,
        "n_distinct": len(counter),
        "entropy": entropy,
        "norm_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "effective_pairs": 2 ** entropy,
        "hhi": sum(p * p for p in shares),
        "counter": counter,
    }


def print_distribution(counter, total, indent="   "):
    for pair, n in counter.most_common():
        bar = "█" * int(round(n / total * 40))
        print(f"{indent}{pair:22s} {n:6d}  {n / total:6.1%}  {bar}")


def analyse(path, group_by=None, show_all=False):
    preds = load_predictions(path)
    labels = get_labels(preds)
    n_labels = len(labels)

    print("=" * 78)
    print(f"📄 {path}")
    print("=" * 78)

    # ---------- 1. 整體集中度 ----------
    stats = concentration([p["best_predicted_pair"] for p in preds], n_labels)

    print(f"\n📊 整體 top-1 集中度  (共 {stats['total']} 筆預測)")
    print(f"   最大宗的一對      : {stats['top_pair']}  →  {stats['top_share']:.2%}")
    print(f"   用到幾種不同的對  : {stats['n_distinct']} / {n_labels}")
    print(f"   有效對數 (2^H)    : {stats['effective_pairs']:.2f}   (1=完全固定, {n_labels}=均勻)")
    print(f"   正規化熵          : {stats['norm_entropy']:.3f}   (0=完全固定, 1=均勻)")
    print(f"   HHI               : {stats['hhi']:.3f}   (1=完全固定, {1 / n_labels:.2f}=均勻)")
    print()
    print_distribution(stats["counter"], stats["total"])

    # ---------- 2. 機率邊際：argmax 是不是在雜訊上決定的 ----------
    margins, top1_probs, spans = [], [], []
    for p in preds:
        vals = sorted(p["predicted_probabilities"].values(), reverse=True)
        margins.append(vals[0] - vals[1])
        top1_probs.append(vals[0])
        spans.append(vals[0] - vals[-1])

    margins.sort()
    mid = len(margins) // 2
    print(f"\n🔬 機率邊際 (top1 - top2)")
    print(f"   中位數 {margins[mid]:.4f} | 平均 {sum(margins) / len(margins):.4f} "
          f"| 最小 {margins[0]:.4f} | 最大 {margins[-1]:.4f}")
    print(f"   top-1 機率範圍   : {min(top1_probs):.3f} ~ {max(top1_probs):.3f}")
    print(f"   單筆內 10 對的全距 (top1-top10) 平均: {sum(spans) / len(spans):.4f}")
    print("   → 邊際遠小於機率本身的量級時，argmax 基本上是在雜訊上做決定")

    # ---------- 3. 同一題跨輸入語言，選擇是否一致 ----------
    by_q = defaultdict(dict)
    for p in preds:
        by_q[(p["dataset"], p["q_id"])][p["input_language"]] = p["best_predicted_pair"]

    n_choices = Counter(len(set(v.values())) for v in by_q.values())
    n_q = len(by_q)
    print(f"\n🌐 同一題在 {max(len(v) for v in by_q.values())} 個輸入語言下的選擇一致性  (共 {n_q} 題)")
    for k in sorted(n_choices):
        print(f"   選出 {k} 種不同的對: {n_choices[k]:5d} 題  ({n_choices[k] / n_q:6.1%})")

    same_rate = n_choices.get(1, 0) / n_q
    if same_rate > 0.9:
        print("   → 幾乎都只選出 1 種，router 的輸出對輸入內容不敏感")
    else:
        print("   → 選擇會隨輸入語言變動；但若機率邊際極小，這只是 argmax 在雜訊上翻動，不是真的在分辨")

    # ---------- 4. 對照：真實的每對勝率 ----------
    ys = {k: None for k in by_q}
    for p in preds:
        ys[(p["dataset"], p["q_id"])] = p["actual_ground_truth_y"]
    y_list = list(ys.values())
    win_rate = [sum(y[i] for y in y_list) / len(y_list) for i in range(n_labels)]
    order = sorted(range(n_labels), key=lambda i: -win_rate[i])

    print(f"\n🎯 對照：各對的真實勝率 (題目層級, n={len(y_list)})")
    for i in order:
        mark = "  ← router 押的那一對" if labels[i] == stats["top_pair"] else ""
        print(f"   {labels[i]:22s} {win_rate[i]:6.1%}{mark}")
    best_i = order[0]
    print(f"\n   最佳固定對 = {labels[best_i]} ({win_rate[best_i]:.2%})")
    print(f"   router 押的 = {stats['top_pair']} ({win_rate[labels.index(stats['top_pair'])]:.2%})")

    # ---------- 5. 分組細看 ----------
    if group_by:
        key = "dataset" if group_by == "dataset" else "input_language"
        print(f"\n📂 依 {key} 分組")
        groups = defaultdict(list)
        for p in preds:
            groups[p[key]].append(p["best_predicted_pair"])

        print(f"\n   {'group':<16} {'n':>6} {'最大宗的對':<22} {'佔比':>8} {'有效對數':>9}")
        print("   " + "-" * 66)
        for g in sorted(groups):
            s = concentration(groups[g], n_labels)
            print(f"   {g:<16} {s['total']:>6} {s['top_pair']:<22} "
                  f"{s['top_share']:>7.1%} {s['effective_pairs']:>9.2f}")
            if show_all:
                print_distribution(s["counter"], s["total"], indent="      ")
                print()

    print()


def main():
    parser = ArgumentParser(description="檢查 router top-1 預測的集中度")
    parser.add_argument("files", nargs="+", help="inference 輸出的 JSON (可用 glob)")
    parser.add_argument("--by", choices=["dataset", "language"],
                        help="額外依 dataset 或 input_language 分組")
    parser.add_argument("--show-all", action="store_true",
                        help="分組時印出完整分佈")
    args = parser.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) or [f])

    for path in paths:
        try:
            analyse(path, group_by=args.by, show_all=args.show_all)
        except Exception as e:
            print(f"❌ {path}: {e}\n")


if __name__ == "__main__":
    main()
