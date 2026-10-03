import os
import json
import glob
import sys
import pandas as pd

LANGUAGE_ORDER = ["english", "chinese", "japanese", "spanish", "russian"]
OUTPUT_DIR = "result/analysis/legacy"


def make_pair_key(langs):
    """依照 LANGUAGE_ORDER 排序兩個語言，組成統一的 pair 字串。"""
    ordered = sorted(langs[:2], key=lambda l: LANGUAGE_ORDER.index(l.lower()) if l.lower() in LANGUAGE_ORDER else 99)
    return f"{ordered[0].capitalize()} vs {ordered[1].capitalize()}"


def generate_challenge_report(dir_path):
    if not os.path.exists(dir_path):
        print(f"❌ 找不到目錄: {dir_path}")
        return

    files = glob.glob(os.path.join(dir_path, "*.json"))
    if not files:
        print(f"⚠️ 在 {dir_path} 找不到任何 JSON 檔案。")
        return

    results = []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 略過檔案 (JSON 格式錯誤): {os.path.basename(filepath)}")
                continue

        if not data or len(data) == 0:
            continue

        meta = data[0] if isinstance(data, list) else data.get("metadata", {})

        if "ExactMatch_Accuracy" not in meta:
            continue

        model_name = meta.get("Model", {}).get("displayName", "Unknown")
        dataset_name = meta.get("Dataset", {}).get("displayName", "Unknown")
        languages = meta.get("Strategy", {}).get("languages", [])
        accuracy = meta.get("ExactMatch_Accuracy", 0.0)

        if len(languages) >= 2:
            pair = make_pair_key(languages)
        elif len(languages) == 1:
            pair = languages[0].capitalize()
        else:
            pair = "Unknown"

        results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Language Pair": pair,
            "Accuracy (%)": round(accuracy * 100, 2),
        })

    if not results:
        print("⚠️ 找不到包含 'ExactMatch_Accuracy' 的檔案。請確認是否已執行過 TestEM。")
        return

    df = pd.DataFrame(results)

    # 排序 Language Pair：依照第一個語言在 LANGUAGE_ORDER 的位置
    all_pairs = df["Language Pair"].unique().tolist()
    def pair_sort_key(pair):
        first = pair.split(" vs ")[0].lower()
        return LANGUAGE_ORDER.index(first) if first in LANGUAGE_ORDER else 99
    ordered_pairs = sorted(all_pairs, key=pair_sort_key)
    df["Language Pair"] = pd.Categorical(df["Language Pair"], categories=ordered_pairs, ordered=True)

    print(f"\n📊 Challenge 實驗成績 Pivot 表 — {dir_path}")
    print("=" * 80)

    markdown_output = ""
    for dataset in sorted(df["Dataset"].unique()):
        sub = df[df["Dataset"] == dataset]
        pivot = sub.pivot_table(
            index="Language Pair",
            columns="Model",
            values="Accuracy (%)",
            aggfunc="first",
        )
        pivot = pivot.sort_index()
        pivot_str = pivot.map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

        md_table = pivot_str.to_markdown(tablefmt="github")
        block = f"### {dataset}\n\n{md_table}\n"
        print(block)
        markdown_output += block + "\n"

    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_md = os.path.join(OUTPUT_DIR, "challenge_results_summary.md")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Challenge 實驗成績 — {dir_path}\n\n" + markdown_output)
    print(f"💾 Markdown 已匯出至: {output_md}")

    output_csv = os.path.join(OUTPUT_DIR, "challenge_results_summary.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 原始資料 CSV 已匯出至: {output_csv}")


if __name__ == "__main__":
    TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "result/challenge"
    generate_challenge_report(TARGET_DIR)
