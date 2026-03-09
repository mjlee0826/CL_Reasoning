import torch
import argparse
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

class ModelPredictor:
    def __init__(self, model_path, device=None):
        """
        model_path: 訓練好的模型路徑 (例如 "./xlmModel/xlm_gpt4omin/checkpoint-500")
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📥 Loading model from {model_path} to {self.device}...")
        
        # 1. 載入模型與 Tokenizer
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        
        # 移動到 GPU (如果有)
        self.model.to(self.device)
        self.model.eval() # 重要！切換到預測模式 (關閉 Dropout)

        # 2. 定義標籤對照表 (順序必須跟訓練時完全一樣！)
        # 在 OnlyDiffDataSpliter 中，我們使用了 sorted(pair_keys)
        # 以下是 5 種語言 (chinese, english, japanese, russian, spanish) 兩兩配對後的字母排序預設猜測
        # ⚠️ 注意：如果你的 JSON 檔中的 key 命名 (例如 english_vs_chinese) 與下方不同，請務必手動調整這裡的字串與順序！
        self.label_names = [
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

        # 防呆檢查：確認模型輸出的維度與標籤數量一致
        num_labels = self.model.config.num_labels
        if len(self.label_names) != num_labels:
            print(f"⚠️ 警告: 模型有 {num_labels} 個輸出，但標籤清單有 {len(self.label_names)} 個！")
            # 動態補齊或截斷標籤名稱以防當機
            self.label_names = [f"Pair_{i}" for i in range(num_labels)]

    def predict(self, text, threshold=0.5):
        # 3. 資料前處理 (Tokenization)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,         # 需與訓練時一致
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 將 tensor 移到 GPU
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 4. 進行預測
        with torch.no_grad(): # 關閉梯度計算，省記憶體並加速
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Logits -> Sigmoid 機率
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 5. 後處理 (解析結果)
        results = {}
        predicted_labels = []
        
        print(f"\n📝 測試題目:\n{text}")
        print("-" * 40)
        print(f"📊 預測結果 (Threshold = {threshold}):")
        
        for idx, prob in enumerate(probs):
            label_name = self.label_names[idx]
            is_active = prob >= threshold
            
            results[label_name] = {
                "probability": float(f"{prob:.4f}"),
                "prediction": 1 if is_active else 0
            }
            
            status = "✅ 推薦使用" if is_active else "❌ 不推薦"
            print(f"{label_name:<20}: {prob:>6.1%} ({status})")
            
            if is_active:
                predicted_labels.append(label_name)
                
        return predicted_labels, results

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-label Model Inference")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("-t", "--text", type=str, help="Question text to predict", default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 初始化預測器
    predictor = ModelPredictor(args.model_path)

    # 決定要測試的問題
    if args.text:
        test_question = args.text
    else:
        # 如果沒有從指令列傳入問題，就使用預設的測試題目
        test_question = "質問があります：\nグル・ナーナクと彼の後継者たちは、以下のうちどれを促進するために特定のシク教ラーガの使用を指示しましたか？\n\n選択肢：\nA: 調和と均衡\nB: 献身と敬虔\nC: 愛と情熱\nD: 善意と愛\n\n質問に基づいて選択肢を選んでください。"

    # 進行預測
    labels, details = predictor.predict(test_question, threshold=args.threshold)
    
    print("-" * 40)
    print(f"🏆 最終建議採用的對戰組合: {labels}")