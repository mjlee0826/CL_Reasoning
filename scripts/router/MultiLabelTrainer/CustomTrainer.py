import torch
from torch import nn
from transformers import Trainer

class ConservativeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # --- 1. 建立遮罩 (Mask) ---
        # 假設 -100 代表缺失資料 (Missing Label)
        # mask 為 1 代表有效，為 0 代表無效
        mask = (labels != -100).float()
        
        # --- 2. 清洗標籤 (Clean Labels) ---
        # BCEWithLogitsLoss 無法計算 -100，會導致 NaN
        # 我們把 -100 暫時變為 0 (反正等一下會被 mask 乘以 0 抵銷掉，改成 1 也可以)
        # .clamp(min=0) 會把所有負數變成 0
        clean_labels = labels.clamp(min=0)
        
        # --- 3. 設定權重 (Pos Weight) ---
        # 這裡設定 0.6 (比 1 小，稍微懲罰誤報，鼓勵保守)
        weights = torch.full((logits.shape[1],), 0.4).to(logits.device) 
        
        # --- 4. 計算原始 Loss (不平均) ---
        # 🔥 關鍵：設定 reduction='none'，這樣它會回傳一個跟 logits 形狀一樣的 loss 矩陣
        # 而不是直接回傳一個平均數
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights, reduction='none')
        loss = loss_fct(logits, clean_labels)
        
        # --- 5. 套用遮罩並計算平均 ---
        # 把無效資料的 Loss 變成 0
        masked_loss = loss * mask
        
        # 算出平均 Loss
        # 分母是「有效資料的總數 (mask.sum())」，加上 1e-9 防止除以零
        final_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        
        return (final_loss, outputs) if return_outputs else final_loss