import random
from File.File import File
from DataSpliter.DataSpliter import DataSpliter

class OnlyDiffDataSpliter(DataSpliter):
    """
    資料切割器：只選用「10個語言配對答案不完全一致」的問題。
    將題目轉化為特徵 X (5 種語言的問題)，並將 10 組語言配對的對錯轉為標籤 y。
    同時保留Dataset 的 Metadata 以供 Inference 效能分析。
    """
    def __init__(self):
        super().__init__()
        self.train_meta = []
        self.val_meta = []
        self.label = [
            'chinese_vs_english',
            'chinese_vs_japanese',
            'chinese_vs_russian',
            'chinese_vs_spanish',
            'english_vs_japanese',
            'english_vs_russian',
            'english_vs_spanish',
            'japanese_vs_russian',
            'japanese_vs_spanish',
            'russian_vs_spanish'
        ]

    def splitData(self, files: list[File], ratio: float) -> tuple:
        X_all = []
        y_all = []
        meta_all = []
        same_status = {}
        
        for file_obj in files:
            # 取得這份檔案對應的 Dataset
            try:
                dataset_name = file_obj.getDatasetConfig().datasetType
            except Exception:
                dataset_name = "UnknownDataset"

            for q_id, item_data in file_obj.records_map.items():
                
                pair_keys = [k for k in item_data.keys() if k != "id"]
                
                if len(pair_keys) < 10:
                    continue
                
                answers = []
                for pair_key in pair_keys:
                    answers.append(item_data[pair_key].get("MyAnswer", ""))
                    
                if len(set(answers)) <= 1:
                    if same_status.get(dataset_name) == None:
                        same_status[dataset_name] = {"total": 0, "correct": 0}

                    same_status[dataset_name]["total"] += 1
                    if item_data[pair_keys[0]]["MyAnswer"] == item_data[pair_keys[0]]["Answer"]:
                        same_status[dataset_name]["correct"] += 1
                    continue
                    
                x_item = {}
                for pair_key in pair_keys:
                    langs = pair_key.split("_vs_")
                    if len(langs) == 2:
                        record = item_data[pair_key]
                        x_item[langs[0]] = record.get("Question1", "")
                        x_item[langs[1]] = record.get("Question2", "")
                        
                if len(x_item) != 5:
                    continue
                    
                y_item = []
                sorted_pairs = sorted(pair_keys)
                
                for pair_key in sorted_pairs:
                    record = item_data[pair_key]
                    my_ans = record.get("MyAnswer", "")
                    correct_ans = record.get("Answer", "")
                    
                    is_correct = 1 if my_ans == correct_ans else 0
                    y_item.append(is_correct)
                    
                X_all.append(x_item)
                y_all.append(y_item)
                
                # 🎯 記錄這筆資料的元數據
                meta_all.append({
                    "dataset": dataset_name,
                    "q_id": q_id
                })
                
        # 綁定 X, y, meta 一起打亂，確保彼此對應關係不變
        combined = list(zip(X_all, y_all, meta_all))
        
        random.seed(42)
        random.shuffle(combined)
        
        split_idx = int(len(combined) * ratio)
        
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        train_X = [item[0] for item in train_data]
        train_y = [item[1] for item in train_data]
        val_X = [item[0] for item in val_data]
        val_y = [item[1] for item in val_data]
        
        # 🎯 偷偷存在 Class 變數中，inference.py 隨時可以拿出來用，且不影響 train.py 的 Tuple 解構
        train_meta = [item[2] for item in train_data]
        val_meta = [item[2] for item in val_data]
        
        print(f"📊 資料切割完成！總計篩選出 {len(combined)} 筆不一致資料。")
        print(f" 🔹 Train data 共 {len(train_X)} 筆")
        print(f" 🔹 Val data 共 {len(val_X)} 筆\n")
        
        return train_X, train_y, val_X, val_y, train_meta, val_meta, same_status