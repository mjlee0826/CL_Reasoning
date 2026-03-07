import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

class ModelPredictor:
    def __init__(self, model_path, device=None):
        """
        model_path: 訓練好的模型路徑 (例如 "./xlm-roberta-multilabel-output/checkpoint-500")
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} to {self.device}...")
        
        # 1. 載入模型與 Tokenizer
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        
        # 移動到 GPU (如果有)
        self.model.to(self.device)
        self.model.eval() # 重要！切換到預測模式 (關閉 Dropout)

        # 定義標籤對照表 (順序必須跟訓練時完全一樣！)
        # 請根據您 DataReader 裡的 strategy_map 順序
        self.label_names = ['Only Chinese', 'Only English', 'Only Spanish', 'Only Japanese', 'Only Russian'] # 範例

    def predict(self, text, threshold=0.5):
        # 2. 資料前處理 (Tokenization)
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

        # 3. 進行預測
        with torch.no_grad(): # 關閉梯度計算，省記憶體並加速
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Logits -> Sigmoid 機率
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 4. 後處理 (解析結果)
        results = {}
        predicted_labels = []
        
        print(f"\n題目: {text}")
        print("-" * 30)
        for idx, prob in enumerate(probs):
            label_name = self.label_names[idx]
            is_active = prob >= threshold
            
            results[label_name] = {
                "probability": float(f"{prob:.4f}"),
                "prediction": 1 if is_active else 0
            }
            
            status = "✅ 答對" if is_active else "❌ 答錯"
            print(f"{label_name:<15}: {prob:.1%} ({status})")
            
            if is_active:
                predicted_labels.append(label_name)
                
        return predicted_labels, results

# --- 使用範例 ---
if __name__ == "__main__":
    # 這裡填入您訓練輸出的資料夾路徑
    # 如果您有設 load_best_model_at_end=True，最好的模型會在 checkpoint 資料夾裡
    # 或者直接指到 output_dir (如果訓練正常結束)
    MODEL_PATH = "./xlm-roberta-multilabel-output6/checkpoint-1500" # 範例路徑，請修改

    # 初始化預測器
    predictor = ModelPredictor(MODEL_PATH)

    # 測試一個新問題
    # new_question = "問題：\nあるクラスの15人の生徒の平均年Есть задача: \nРаботу, которую можно было бы закончить за 7 дней, закончили на 3 дня раньше после того, как присоединилось ещё 10 человек. Каково было количество занятых рабочих?.\nИ есть 5 вариантов ответа\na ) 22 , b ) 20 , c ) 60 , d ) 71 , e ) 11\nПожалуйста, выберите вариант, основываясь на вопросе.\nВ конце вашего ответа предоставьте его в точности в этом JSON-формате:\n{\"answer\": \"your_letter_choice\"}\nОтвет должен быть одной английской буквой (a-e). Вы должны использовать двойные кавычки. Вы должны вывести только одну строку."
    # 進行預測 (閾值可以設高一點，例如 0.75，來過濾沒信心的)
    # new_question = "Hay una Pregunta:  \nUn grano de arena podría causarte molestias si se mete en tu ¿qué?  \n\nY hay 5 opciones:  \nA) cubo, B) montón de arena, C) playa, D) almeja, E) traje de baño  \n\nPor favor, elige una opción basándote en la pregunta.  \nAl final de tu respuesta, proporciona tu respuesta en este formato JSON exacto:  \n{\"answer\": \"tu_letra_elegida\"}  \nLa respuesta debe ser una sola letra en inglés (A-E). Debes usar comillas dobles. Debes generar solo una línea."
    new_question = "質問があります：\nグル・ナーナクと彼の後継者たちは、以下のうちどれを促進するために特定のシク教ラーガの使用を指示しましたか？\n\n選択肢：\nA: 調和と均衡\nB: 献身と敬虔\nC: 愛と情熱\nD: 善意と愛\n\n質問に基づいて選択肢を選んでください。\n回答の最後に、以下の正確なJSON形式で答えを提供してください：\n{\"answer\": \"your_letter_choice\"}\n答えは単一の英字（A-D）でなければなりません。二重引用符を出力する必要があります。出力は一行のみである必要があります。"
    labels, details = predictor.predict(new_question, threshold=0.5)
    
    print("\n最終推薦策略:", labels)
