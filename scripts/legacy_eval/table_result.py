import os
import json
import glob
import pandas as pd

LANGUAGE_ORDER = ["english", "chinese", "japanese", "spanish", "russian"]
OUTPUT_DIR = "result/analysis/legacy"

def generate_report(dir_path):
    """
    掃描指定資料夾內的所有 JSON 檔，萃取 Metadata 中的實驗成績，
    以 Dataset 為單位輸出 pivot 表格（Row: Language, Column: Model）。
    """
    if not os.path.exists(dir_path):
        print(f"❌ 找不到目錄: {dir_path}")
        return

    search_pattern = os.path.join(dir_path, "*.json")
    files = glob.glob(search_pattern)

    if not files:
        print(f"⚠️ 在 {dir_path} 找不到任何 JSON 檔案。")
        return

    results = []

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 略過檔案 (JSON 格式錯誤): {os.path.basename(filepath)}")
                continue

            if not data or len(data) == 0:
                continue

            meta = data[0]

            if "ExactMatch_Accuracy" not in meta:
                continue

            model_name = meta.get("Model", {}).get("displayName", "Unknown")
            dataset_name = meta.get("Dataset", {}).get("displayName", "Unknown")
            language = meta.get("Strategy", {}).get("languages", ["Unknown"])[0]
            accuracy = meta.get("ExactMatch_Accuracy", 0.0)

            results.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "Language": language.capitalize(),
                "Accuracy (%)": round(accuracy * 100, 2),
            })

    if not results:
        print("⚠️ 找不到包含 'ExactMatch_Accuracy' 成績的檔案。請確認是否已執行過 TestEM。")
        return

    df = pd.DataFrame(results)

    # 語言排序：依照 LANGUAGE_ORDER，其餘按字母排
    lang_order = [l.capitalize() for l in LANGUAGE_ORDER]
    all_langs = df["Language"].unique().tolist()
    ordered_langs = [l for l in lang_order if l in all_langs] + \
                    sorted([l for l in all_langs if l not in lang_order])
    df["Language"] = pd.Categorical(df["Language"], categories=ordered_langs, ordered=True)

    print(f"\n📊 實驗成績 Pivot 表 — {dir_path}")
    print("=" * 80)

    markdown_output = ""
    for dataset in sorted(df["Dataset"].unique()):
        sub = df[df["Dataset"] == dataset]
        pivot = sub.pivot_table(
            index="Language",
            columns="Model",
            values="Accuracy (%)",
            aggfunc="first"
        )
        pivot = pivot.sort_index()
        pivot_str = pivot.map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

        md_table = pivot_str.to_markdown(tablefmt="github")
        block = f"### {dataset}\n\n{md_table}\n"
        print(block)
        markdown_output += block + "\n"

    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_md = os.path.join(OUTPUT_DIR, "experiment_results_summary.md")
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(f"# 實驗成績 — {dir_path}\n\n" + markdown_output)
    print(f"💾 Markdown 已匯出至: {output_md}")

    output_csv = os.path.join(OUTPUT_DIR, "experiment_results_summary.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 原始資料 CSV 已匯出至: {output_csv}")

if __name__ == "__main__":
    import sys                                                                                                                                                                                                                        
    TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "result/self_reflection/"
    generate_report(TARGET_DIR)