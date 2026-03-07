from argparse import ArgumentParser

from Model.ModelType import MODEL_LIST
from Dataset.DatasetType import DATASET_LIST
from Strategy.StrategyType import STRATEGY_LIST

from MultiLabelTrainer.DataReader import DataReader
from MultiLabelTrainer.MultiLabelDataset import MultiLabelDataset
from MultiLabelTrainer.Metric import compute_metrics
from MultiLabelTrainer.CustomTrainer import ConservativeTrainer

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

def parseArgs():
    parser = ArgumentParser()
    
    parser.add_argument("-m", "--model", nargs="+", choices=MODEL_LIST, help="choose your model")
    parser.add_argument("-d", "--dataset", nargs="+", choices=DATASET_LIST, help="choose your dataset")
    parser.add_argument("-s", "--strategy", nargs="+", choices=STRATEGY_LIST, help="choose your strategy")
    parser.add_argument("--dirpath", help="your dir path")
    parser.add_argument("--split", default=0.7, type=float, help="split train - val")
    parser.add_argument("--datanums", default=1000, type=int, help="total data nums")
    parser.add_argument("--maxlens", default=512, type=int, help="data max length")

    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    
    (train_texts, train_labels), (val_texts, val_labels) = DataReader(args.dirpath, args.model, args.dataset, args.strategy).getDataset(args.datanums, args.split)
    count = 0
    for label in train_labels:
        if label == [1, 1, 1, 1, 1]:
            coumt += 1
    print(f'{count} / {len(train_labels)}')
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, args.maxlens)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer, args.maxlens)
    
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(args.strategy), 
        problem_type="multi_label_classification" 
    )

    args = TrainingArguments(
        output_dir="xlm-roberta-multilabel-output6",
        eval_strategy="steps",    # 🔥 改成 steps
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        learning_rate=1e-5,
        num_train_epochs=20,          # 🔥 直接設大一點 (例如 20)
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        fp16=True,
        weight_decay=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # 多標籤通常看 F1-Micro
        greater_is_better=False,

        logging_strategy="steps",
        logging_steps=50,  # 每 10 步紀錄一次 Training Loss (畫圖比較平滑)
        report_to="tensorboard"   # 先設 none，我們下面手動用 Matplotlib 畫圖
    )

    trainer = ConservativeTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
    )

    # 開始訓練
    trainer.train()


if __name__ == '__main__':
    main()