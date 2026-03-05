import os
import json
import glob

def migrate_old_format(input_dir="result/tempature0", output_dir="new_english_result"):
    """
    Migrates old format evaluation JSON files to the new V2 architecture format.
    - Reconstructs the index 0 Metadata into nested Config dictionaries.
    - Injects an 'id' field into every data record for O(1) alignment in the new framework.
    - Limits the maximum number of records to 2000.
    """
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 輸出目錄已準備就緒: {output_dir}/")

    # 尋找所有結尾是 English.json 的檔案
    search_pattern = os.path.join(input_dir, "*English.json")
    files = glob.glob(search_pattern)

    if not files:
        print("❌ 在指定目錄下找不到任何符合的 JSON 檔案。")
        return

    print(f"🔍 找到 {len(files)} 個檔案，準備開始轉檔...\n")

    success_cnt = 0
    MAX_RECORDS = 2000  # 設定最大資料筆數限制

    for filepath in files:
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)

        # 讀取舊資料
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                old_data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ 略過檔案 (JSON 格式錯誤): {filename}")
                continue

        if len(old_data) < 2:
            print(f"⚠️ 略過檔案 (資料不完整): {filename}")
            continue

        # 取出舊的 Meta 與 Records
        old_meta = old_data[0]
        records = old_data[1:]

        # --- ⭐ 新增功能：若資料超過 2000 筆，則只取前 2000 筆 ---
        if len(records) > MAX_RECORDS:
            records = records[:MAX_RECORDS]
            print(f"   ✂️ 檔案 {filename} 資料超過 {MAX_RECORDS} 筆，已自動截斷。")

        # --- 1. 構建新的 Model Config ---
        raw_model = old_meta.get("Model", "Unknown")
        # 產生 modelType: 例如 "GPT 4.1 mini" 變成 "gpt4.1mini"
        model_type = raw_model.replace(" ", "").lower() 
        new_model_config = {
            "modelType": model_type,
            "temperature": 0.0,  # 來自 tempature0 資料夾的設定
            "modelName": raw_model,
            "displayName": raw_model
        }

        # --- 2. 構建新的 Dataset Config ---
        raw_dataset = old_meta.get("Dataset", "Unknown")
        # 產生 datasetType: 例如 "CMB-Exam" 變成 "cmbexam"
        dataset_type = raw_dataset.replace("-", "").lower()
        if dataset_type == "commenseqa": # 處理舊版可能的拼字問題
            dataset_type = "commonsenseqa"
            
        nums = old_meta.get("Data Nums", -1)
        samples = old_meta.get("Data Samples", 1)
        
        # ⭐ 同步修正 Meta 中的數量設定，確保 Config 與實際資料長度一致
        if nums > MAX_RECORDS:
            nums = MAX_RECORDS
        elif nums == -1 and len(records) == MAX_RECORDS:
            nums = MAX_RECORDS
        
        new_dataset_config = {
            "datasetType": dataset_type,
            "displayName": raw_dataset,
            "nums": nums,
            "sample": samples,
            "language": "english",
            "dataNums": nums * samples if nums != -1 else len(records)
        }

        # --- 3. 構建新的 Strategy Config ---
        new_strategy_config = {
            "strategyType": "onelanguage",
            "languages": ["english"],
            "displayName": "One Language (english)"
        }

        # --- 4. 組合新的資料結構 ---
        new_data = [
            {
                "Model": new_model_config,
                "Dataset": new_dataset_config,
                "Strategy": new_strategy_config
            }
        ]

        # --- 5. 注入 id 欄位 ---
        # 為了支援新版的 ResultFile.py 中以 id 為 key 的 Hash Map 機制
        for idx, rec in enumerate(records):
            new_record = {"id": idx}
            new_record.update(rec) # 將原本的 Question, Translated, Result 等欄位合併進來
            new_data.append(new_record)

        # --- 6. 寫入新檔案 ---
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 成功轉換: {filename}")
        success_cnt += 1

    print(f"\n🎉 轉檔完成！共成功轉換 {success_cnt} 個檔案。")

if __name__ == "__main__":
    migrate_old_format()