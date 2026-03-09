import sys
from argparse import ArgumentParser

from File.FileFactory import FileFactory
from DataSpliter.OnlyDiffDataSpliter import OnlyDiffDataSpliter

from MultiLabelTrainer.MultiLabelDataset import MultiLabelDataset
from MultiLabelTrainer.Metric import compute_metrics
from MultiLabelTrainer.CustomTrainer import ConservativeTrainer

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, EarlyStoppingCallback

def parseArgs():
    parser = ArgumentParser(description="Multi-Label Classification Training Script")
    
    # 改為接收 Transform 過後的資料夾路徑
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the TRANSFORMED training JSON files")
    parser.add_argument("-e", "--extension", default="*.json", help="File extension to look for")
    parser.add_argument("--output_dir", default="xlm-roberta-multilabel-output", help="Directory to save model checkpoints")
    
    parser.add_argument("--split", default=0.7, type=float, help="split ratio for train - val")
    parser.add_argument("--maxlens", default=512, type=int, help="Tokenizer max length")
    parser.add_argument("--batch_size", default=64, type=int, help="Training and Evaluation Batch Size")
    parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs")

    args = parser.parse_args()
    return args

def expand_data(X_data, y_data):
    """
    將一筆包含 5 種語言的 Dictionary，拆解為 5 筆獨立的訓練資料。
    y 標籤也會同步複製 5 份，讓 X 和 y 數量保持一致。
    """
    expanded_texts = []
    expanded_labels = []
    
    for x_dict, y_item in zip(X_data, y_data):
        # x_dict 裡面有 5 個 key-value (語言: 問題字串)
        for lang, question in x_dict.items():
            # 將各語言的題目當作獨立的一筆輸入
            expanded_texts.append(question)
            # 對應的標籤 y 也要跟著複製一份
            expanded_labels.append(y_item)
            
    return expanded_texts, expanded_labels

def main():
    args = parseArgs()
    
    print(f"🚀 準備載入訓練資料 (Loading files from: {args.input_dir})...")
    
    # 1. 讀取 Transform 過後的檔案
    file_factory = FileFactory()
    files = file_factory.getFileInDir(args.input_dir, extension=args.extension)
    
    if not files:
        print(f"❌ 錯誤: 在目錄 {args.input_dir} 中找不到任何 {args.extension} 檔案。")
        sys.exit(1)
        
    print(f"📄 成功載入 {len(files)} 個檔案，準備進行資料切割 (Splitting data)...")
    
    # 2. 進行資料切割 (只取答案不一致的資料)
    spliter = OnlyDiffDataSpliter()
    train_X, train_y, val_X, val_y, _, _, _ = spliter.splitData(files, args.split)
    
    if not train_X:
        print("❌ 錯誤: 篩選後沒有足夠的資料可供訓練！請檢查資料集是否符合條件。")
        sys.exit(1)
        
    # 3. 將資料擴展 (1 筆拆為 5 筆)
    train_texts, train_y = expand_data(train_X, train_y)
    val_texts, val_y = expand_data(val_X, val_y)
    
    print(f"📈 擴展後資料量: Train data 共 {len(train_texts)} 筆, Val data 共 {len(val_texts)} 筆")

    # 4. 統計資料
    count = 0
    # 動態判斷全對的條件 (因為 y 的長度可能會變)
    full_correct_length = len(train_y[0]) 
    for label in train_y:
        # 如果該陣列加總等於它的長度，代表全都是 1
        if sum(label) == full_correct_length:
            count += 1
            
    print(f'🎯 [統計] Train Set 中全對 (All 1s) 的比例: {count} / {len(train_y)}')
    
    num_labels = len(train_y[0])
    print(f"🏷️ 偵測到模型預測標籤數量 (num_labels): {num_labels}")

    # 5. 建立 Dataset 與 Tokenizer
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    train_dataset = MultiLabelDataset(train_texts, train_y, tokenizer, args.maxlens)
    val_dataset = MultiLabelDataset(val_texts, val_y, tokenizer, args.maxlens)
    
    # 6. 建立模型
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        problem_type="multi_label_classification" 
    )

    # 7. 設定 Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",    
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        learning_rate=1e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        weight_decay=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,  
        report_to="tensorboard"   
    )

    # 8. 綁定自定義的 ConservativeTrainer
    trainer = ConservativeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
    )

    # 9. 開始訓練
    print("\n🔥 開始訓練 (Start Training)...")
    trainer.train()

if __name__ == '__main__':
    main()