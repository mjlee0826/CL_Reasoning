import os
import json
import glob
import pandas as pd

def generate_report(dir_path):
    """
    掃描指定資料夾內的所有 JSON 檔，萃取 Metadata 中的實驗成績，
    並輸出成一個排版精美的表格。
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
                
            meta = data[0]
            
            # 確保這個檔案已經被 TestEM 算過成績了
            if "ExactMatch_Accuracy" not in meta:
                continue

            # 安全地萃取所需資訊
            model_name = meta.get("Model", {}).get("displayName", "Unknown")
            dataset_name = meta.get("Dataset", {}).get("displayName", "Unknown")
            # 嘗試從 Strategy 拿語言，若沒有則從 Dataset 拿
            language = meta.get("Strategy", {}).get("languages", ["Unknown"])[0] 
            strategy = meta.get("Strategy", {}).get("strategyType", "Unknown")
            
            correct = meta.get("ExactMatch_Correct", 0)
            total = meta.get("ExactMatch_Total", 0)
            accuracy = meta.get("ExactMatch_Accuracy", 0.0)

            results.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "Language": language.capitalize(),
                "Strategy": strategy.capitalize(),
                "Correct / Total": f"{correct} / {total}",
                "Accuracy (%)": accuracy * 100  # 轉為百分比
            })

    if not results:
        print("⚠️ 找不到包含 'ExactMatch_Accuracy' 成績的檔案。請確認是否已執行過 TestEM。")
        return

    # 1. 轉換成 Pandas DataFrame
    df = pd.DataFrame(results)

    # 2. 依照 Dataset -> Language -> Model 進行排序，讓表格看起來更有條理
    df = df.sort_values(by=["Dataset", "Language", "Model"]).reset_index(drop=True)

    # 3. 格式化小數點 (只保留兩位小數)
    df["Accuracy (%)"] = df["Accuracy (%)"].apply(lambda x: f"{x:.2f}%")

    # 4. 在終端機印出漂亮表格 (需安裝 tabulate)
    print("\n📊 實驗成績彙總表 (Experiment Results Summary):")
    print("=" * 80)
    print(df.to_markdown(index=False, tablefmt="github"))
    print("=" * 80)

    # 5. (可選) 匯出成 CSV，方便你用 Excel 打開
    output_csv = "experiment_results_summary.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 表格已自動匯出至: {output_csv}")

if __name__ == "__main__":
    # 你可以把這裡換成你實際存放 JSON 的資料夾路徑
    TARGET_DIR = "result/baseline" 
    generate_report(TARGET_DIR)