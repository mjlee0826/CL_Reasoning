import os
import json
import glob
import pandas as pd

def generate_challenge_report(dir_path):
    """
    掃描指定資料夾內的所有 JSON 檔，萃取 Metadata 中的實驗成績，
    針對 Challenge 策略，將 Dataset 放上方、Language pair 放左側，
    按 Model 輸出成 HackMD 支援的 Markdown 表格格式。
    """
    if not os.path.exists(dir_path):
        print(f"❌ 找不到目錄: {dir_path}")
        return

    # 尋找所有 json 檔案
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
            
            # 確保檔案有內容
            if not data or len(data) == 0:
                continue
                
            # 兼容處理：有時 metadata 放在 list 第一個元素，有時放在 dict 的 "metadata" 鍵中
            if isinstance(data, list):
                meta = data[0]
            else:
                meta = data.get("metadata", {})
            
            # 確保這個檔案已經被 TestEM 算過成績了
            if "ExactMatch_Accuracy" not in meta:
                continue

            # 安全地萃取所需資訊
            model_name = meta.get("Model", {}).get("displayName", "Unknown")
            dataset_name = meta.get("Dataset", {}).get("displayName", "Unknown")
            
            # 🎯 取得兩個語言，組成 Language Pair
            languages = meta.get("Strategy", {}).get("languages", ["Unknown"])
            if len(languages) >= 2:
                language_pair = f"{languages[0].capitalize()} vs {languages[1].capitalize()}"
            elif len(languages) == 1:
                language_pair = languages[0].capitalize()
            else:
                language_pair = "Unknown"
                
            accuracy = meta.get("ExactMatch_Accuracy", 0.0)

            results.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "Language Pair": language_pair,
                "Accuracy (%)": accuracy * 100  
            })

    if not results:
        print("⚠️ 找不到包含 'ExactMatch_Accuracy' 成績的檔案。請確認是否已執行過 TestEM。")
        return

    # 1. 轉換成 Pandas DataFrame
    df = pd.DataFrame(results)

    # 2. 格式化小數點為字串 (只保留兩位小數並加上 %)
    df["Accuracy Str"] = df["Accuracy (%)"].apply(lambda x: f"{x:.2f}%")

    # 3. 取得所有不重複的 Model 列表，並排序
    models = sorted(df["Model"].unique())

    print("\n" + "="*80)
    print("📋 請直接複製以下 Markdown 語法貼上至 HackMD：\n")

    markdown_output = ""

    for model in models:
        # 篩選出該 Model 的數據
        model_df = df[df["Model"] == model]

        # 🎯 核心改動：使用 pivot_table 將 Dataset 轉為 Column，Language Pair 轉為 Row
        pivot_df = model_df.pivot_table(
            index="Language Pair", 
            columns="Dataset", 
            values="Accuracy Str", 
            aggfunc="first"
        ).fillna("-").reset_index() # fillna("-") 用來處理如果某個資料集剛好漏了某個語言的防呆

        # 確保 Language Pair 標題正確，並產生 Markdown 表格
        md_table = pivot_df.to_markdown(index=False, tablefmt="github")

        # 組合 Markdown 內容
        markdown_output += f"### 🤖 Model: {model}\n\n"
        markdown_output += md_table + "\n\n<br>\n\n"

    # 在終端機印出所有 Markdown 語法供使用者直接複製
    print(markdown_output)
    print("="*80)

    # 4. 將 Markdown 語法存成文字檔，方便備用
    output_md = "challenge_results_hackmd.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(markdown_output)
    print(f"\n💾 Markdown 語法也已自動存成檔案至: {output_md}")

    # 5. 保留一份原始的一維 CSV 給你
    output_csv = "challenge_results_raw.csv"
    df.drop(columns=["Accuracy Str"]).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 原始分析資料 (CSV) 存至: {output_csv}\n")

if __name__ == "__main__":
    # 將目標路徑指向存放 Challenge 算分結果的資料夾
    TARGET_DIR = "result/challenge" 
    generate_challenge_report(TARGET_DIR)