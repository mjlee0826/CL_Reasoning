from File.File import File
import json

class DataTransform():
    """
    用來將多個 Challenge (辯論) 策略的結果檔案，
    轉換並合併為適合用於訓練 (Training Data) 的 JSON 格式。
    將同一個問題的不同語言對戰結果，整合成一筆資料。
    """
    def __init__(self, files: list[File]):
        self.files = files

    def _transform_data(self) -> list[dict]:
        """
        核心轉換邏輯：
        以 (Dataset名稱, 問題ID) 為 Key 進行分群。
        每一筆轉換後的資料 (dict) 都代表「同一個問題」，
        裡面包含該問題在不同語言組合 (例如 english_vs_chinese) 下的對戰紀錄。
        """
        # grouped_data 結構: { (dataset_name, q_id): { "english_vs_chinese": record, "english_vs_japanese": record, ... } }
        grouped_data = {}

        for file in self.files:
            # 取得檔案中的語言設定
            langs = file.getLanguage()
            
            if not langs or len(langs) < 2:
                print(f"⚠️ 警告: 檔案 {file.file_path} 的語言設定不足兩個，已略過。")
                continue
            
            # 動態生成 Key，例如 "english_vs_chinese"
            lang_pair_key = f"{langs[0]}_vs_{langs[1]}"
            
            # 取得 Dataset 名稱，避免不同資料集有相同的 q_id 互相覆蓋
            try:
                dataset_name = file.getDatasetConfig().datasetType
            except Exception:
                dataset_name = "unknown_dataset"
            
            # 將檔案中的每一題塞入對應的群組
            for q_id, record in file.records_map.items():
                composite_key = (dataset_name, q_id)
                
                if composite_key not in grouped_data:
                    grouped_data[composite_key] = {}
                
                # 將這題的這組語言對戰紀錄存進去
                grouped_data[composite_key][lang_pair_key] = record

        # 將分群好的 dict 轉換成 List，並依照 dataset 和 q_id 進行排序
        # 為了安全排序，如果 q_id 是數字字串則轉為 int 排序，否則照字串排
        def sort_key(k):
            dataset_str = str(k[0])
            id_val = int(k[1]) if str(k[1]).isdigit() else str(k[1])
            return (dataset_str, id_val)

        result_list = [
            grouped_data[key] 
            for key in sorted(grouped_data.keys(), key=sort_key)
        ]
        
        return result_list

    def export_to_json(self, output_path: str):
        """
        將轉換後的資料輸出為實體的 JSON 檔案。
        """
        transformed_data = self._transform_data()
        
        if not transformed_data:
            print("⚠️ 警告：轉換後的資料為空，請檢查輸入的檔案是否正確。")
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=4)
            
        # 計算統計數據
        total_questions = len(transformed_data)
        # 抽樣檢查第一筆資料有幾個語言組合
        sample_pairs_count = len(transformed_data[0].keys())
        
        print(f"🎉 轉換成功！共收集了 {total_questions} 個不重複的問題 (跨 Dataset)。")
        print(f"💡 每個問題平均包含約 {sample_pairs_count} 種語言組合的對戰紀錄。")
        print(f"📁 檔案已儲存至: {output_path}")