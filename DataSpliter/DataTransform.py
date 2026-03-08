import os
import json
from File.File import File

class DataTransform():
    """
    用來將多個 Challenge (辯論) 策略的結果檔案，
    轉換並合併為適合用於訓練 (Training Data) 的 JSON 格式。
    會自動依照 Model 和 Dataset 進行分群，並各自輸出獨立的檔案。
    """
    def __init__(self, files: list[File]):
        self.files = files

    def _group_files(self) -> dict:
        """
        將所有的檔案依照 (ModelType, DatasetType) 進行分群。
        回傳: { (model_name, dataset_name): [file1, file2, ...] }
        """
        groups = {}
        for file in self.files:
            try:
                model_name = file.getModelConfig().modelType
                dataset_name = file.getDatasetConfig().datasetType
            except Exception as e:
                print(f"⚠️ 無法取得檔案 {file.file_path} 的 Model/Dataset 設定: {e}，已略過。")
                continue
            
            key = (model_name, dataset_name)
            if key not in groups:
                groups[key] = []
            groups[key].append(file)
            
        return groups

    def _transform_group(self, group_files: list[File]) -> list:
        """
        針對同一組 (同 Model、同 Dataset) 的檔案集合進行轉換邏輯。
        """
        if not group_files:
            return []

        # 1. 建立 Metadata (放在陣列最前端)
        first_file = group_files[0]
        model_config = first_file.metadata.get("Model", {})
        dataset_config = first_file.metadata.get("Dataset", {})
        
        # 自行建立 Strategy 配置
        strategy_config = {
            "strategyType": "challenge",
            "displayName": "all language pair challenge",
            "languages": [] 
        }

        metadata = {
            "Model": model_config,
            "Dataset": dataset_config,
            "Strategy": strategy_config
        }

        # grouped_data 結構: { q_id: { "id": q_id, "english_vs_chinese": record, ... } }
        grouped_data = {}

        for file in group_files:
            langs = file.getLanguage()
            
            if not langs or len(langs) < 2:
                print(f"⚠️ 警告: 檔案 {file.file_path} 的語言設定不足兩個，已略過。")
                continue
            
            lang_pair_key = f"{langs[0]}_vs_{langs[1]}"
            
            # 直接用 q_id 當作 Key 進行分群 (同 dataset 底下 q_id 是唯一的)
            for q_id, record in file.records_map.items():
                if q_id not in grouped_data:
                    # ✅ 補上 "id" 鍵，讓 File.py 讀取時可以自動建立 records_map
                    grouped_data[q_id] = {"id": q_id}
                
                # 將這題的這組語言對戰紀錄存進去
                grouped_data[q_id][lang_pair_key] = record

        # 將分群好的 dict 轉換成 List，並依照 q_id 進行排序
        def sort_key(k):
            return int(k) if str(k).isdigit() else str(k)

        result_list = [
            grouped_data[key] 
            for key in sorted(grouped_data.keys(), key=sort_key)
        ]
        
        # ✅ 回傳 List 結構：第一筆是 metadata，後面跟著所有的 dict
        return [metadata] + result_list

    def export_to_dir(self, output_dir: str):
        """
        將轉換後的資料，依照 Model 和 Dataset 分群，
        並分別輸出到指定的資料夾中。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        groups = self._group_files()
        
        if not groups:
            print("⚠️ 警告：沒有找到任何有效的檔案組合可以轉換。")
            return

        print(f"🔍 分析完畢，共找到 {len(groups)} 組 (Model, Dataset) 組合，準備分別輸出...")

        for (model_name, dataset_name), group_files in groups.items():
            transformed_data = self._transform_group(group_files)
            
            if not transformed_data or len(transformed_data) <= 1:
                print(f"⚠️ 警告：組合 [{model_name} - {dataset_name}] 轉換後的資料為空，已略過。")
                continue

            output_filename = f"{model_name}_{dataset_name}_challenge_training.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_data, f, ensure_ascii=False, indent=4)
                
            total_questions = len(transformed_data) - 1
            # 計算語言對數時，將 "id" 濾掉
            sample_pairs_count = len([k for k in transformed_data[1].keys() if k != "id"])
            
            print(f"🎉 轉換成功！[{model_name} - {dataset_name}] 共收集了 {total_questions} 個不重複的問題。")
            print(f"💡 每個問題平均包含約 {sample_pairs_count} 種語言組合的對戰紀錄。")
            print(f"📁 檔案已儲存至: {output_path}\n")