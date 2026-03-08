import random
from File.File import File
from DataSpliter.DataSpliter import DataSpliter

class OnlyDiffDataSpliter(DataSpliter):
    """
    資料切割器：只選用「10個語言配對答案不完全一致」的問題。
    將題目轉化為特徵 X (5 種語言的問題)，並將 10 組語言配對的對錯轉為標籤 y。
    """
    def __init__(self):
        super().__init__()

    def splitData(self, files: list[File], ratio: float) -> tuple:
        X_all = []
        y_all = []
        
        for file_obj in files:
            # ✅ 現在可以直接利用 File 類別內建的 records_map，不需要手動用 json.load 讀檔了！
            for q_id, item_data in file_obj.records_map.items():
                
                # 過濾掉 "id" 這個輔助 key，只保留那 10 個語言配對的 keys
                pair_keys = [k for k in item_data.keys() if k != "id"]
                
                # 1. 確保該問題具備完整的 10 個語言配對
                if len(pair_keys) < 10:
                    continue
                
                # 2. 檢查 10 個答案是否「不完全一致」
                answers = []
                for pair_key in pair_keys:
                    answers.append(item_data[pair_key].get("MyAnswer", ""))
                    
                # 如果 10 個答案全數相同 (無分歧)，則捨棄這筆資料
                if len(set(answers)) <= 1:
                    continue
                    
                # 3. 提取 X: 取出五個語言的問題本身
                # 結構範例: {"english": "Question in English", "chinese": "Question in Chinese", ...}
                x_item = {}
                for pair_key in pair_keys:
                    # 解析 pair_key (例如 "english_vs_chinese")
                    langs = pair_key.split("_vs_")
                    if len(langs) == 2:
                        record = item_data[pair_key]
                        x_item[langs[0]] = record.get("Question1", "")
                        x_item[langs[1]] = record.get("Question2", "")
                        
                # 確保成功提取到 5 種不同的語言
                if len(x_item) != 5:
                    continue
                    
                # 4. 提取 y: 長度為 10 的 list，每個元素為 0 (錯) 或 1 (對)
                y_item = []
                
                # 將 pair_keys 排序，保證每筆資料的 y 陣列語言順序永遠一致
                # 字母排序後永遠會是: chinese_vs_english, chinese_vs_japanese, english_vs_japanese... 等固定順序
                sorted_pairs = sorted(pair_keys)
                
                for pair_key in sorted_pairs:
                    record = item_data[pair_key]
                    my_ans = record.get("MyAnswer", "")
                    correct_ans = record.get("Answer", "")
                    
                    # 答對給 1，答錯給 0
                    is_correct = 1 if my_ans == correct_ans else 0
                    y_item.append(is_correct)
                    
                # 將整理好的 X, y 加入總集
                X_all.append(x_item)
                y_all.append(y_item)
                
        # 5. 將過濾後的資料根據 ratio 進行切割
        combined = list(zip(X_all, y_all))
        
        # 設定固定 seed 並打亂資料，確保實驗的可重現性
        random.seed(42)
        random.shuffle(combined)
        
        split_idx = int(len(combined) * ratio)
        
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        train_X = [item[0] for item in train_data]
        train_y = [item[1] for item in train_data]
        val_X = [item[0] for item in val_data]
        val_y = [item[1] for item in val_data]
        
        # 輸出 Train 以及 Val 的資料筆數
        print(f"📊 資料切割完成！總計篩選出 {len(combined)} 筆不一致資料。")
        print(f" 🔹 Train data 共 {len(train_X)} 筆")
        print(f" 🔹 Val data 共 {len(val_X)} 筆\n")
        
        return train_X, train_y, val_X, val_y