from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from DataSpliter.DataSpliter import DataSpliter
import torch

class ModelPredictor:
    def __init__(self, model_path, spliter: DataSpliter, device=None):
        """
        model_path: 訓練好的模型路徑
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📥 Loading model from {model_path} to {self.device}...")
        
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval() 

        self.label_names = spliter.getLabel()

        num_labels = self.model.config.num_labels
        if len(self.label_names) != num_labels:
            print(f"⚠️ 警告: 模型有 {num_labels} 個輸出，但標籤清單有 {len(self.label_names)} 個！")
            self.label_names = [f"Pair_{i}" for i in range(num_labels)]

    def predict(self, text) -> dict:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,         
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad(): 
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        probs_dict = {
            self.label_names[idx]: float(f"{prob:.4f}") 
            for idx, prob in enumerate(probs)
        }
                
        return probs_dict