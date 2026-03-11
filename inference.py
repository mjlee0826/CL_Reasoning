import sys
import os
import json
import argparse
from tqdm import tqdm

from File.FileFactory import FileFactory
from DataSpliter.OnlyDiffDataSpliter import OnlyDiffDataSpliter
from MultiLabelTrainer.ModelPredictor import ModelPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-label Model Validation Inference")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the TRANSFORMED training JSON files")
    parser.add_argument("-e", "--extension", default="*.json", help="File extension to look for")
    parser.add_argument("--split", default=0.7, type=float, help="Split ratio used during training")
    parser.add_argument("-o", "--output_file", required=True, help="Path to save the inference results")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 讀取與切割資料
    print(f"🚀 準備載入驗證資料 (Loading files from: {args.input_dir})...")
    file_factory = FileFactory()
    files = file_factory.getFileInDir(args.input_dir, extension=args.extension)
    
    if not files:
        print(f"❌ 錯誤: 找不到任何 {args.extension} 檔案。")
        sys.exit(1)
        
    spliter = OnlyDiffDataSpliter()
    _, _, val_X, val_y, _, val_meta, same_status = spliter.splitData(files, args.split)
    
    print(f"📄 成功取得 Validation Set: 共 {len(val_X)} 個問題 (將展開為 {len(val_X)*5} 筆測試)")

    # 2. 初始化預測器
    predictor = ModelPredictor(args.model_path, spliter)

    # 3. 進行預測與比對
    final_results = []
    
    print("\n🔍 開始進行預測與比對...")
    
    # 同時打包 val_X, val_y 以及 val_meta
    for x_dict, y_list, meta in tqdm(zip(val_X, val_y, val_meta), total=len(val_X), desc="Predicting Validation Set"):
        
        # 遍歷該問題的 5 種語言版本
        for lang, question_text in x_dict.items():
            
            probs_dict = predictor.predict(question_text)
            best_pair = max(probs_dict, key=probs_dict.get)
            best_pair_idx = predictor.label_names.index(best_pair)
            is_correct = bool(y_list[best_pair_idx] == 1)
            
            # 🎯 儲存結果，並附上 Dataset 以及 Question ID 資訊
            final_results.append({
                "dataset": meta["dataset"],
                "q_id": meta["q_id"],
                "input_language": lang,
                "question": question_text,
                "best_predicted_pair": best_pair,
                "is_correct_in_ground_truth": is_correct,
                "predicted_probabilities": probs_dict,
                "actual_ground_truth_y": y_list 
            })

    # 4. 🎯 進行詳細的 Performance 成效分析 (Group by Dataset, Language)
    diff_stats = {}
    
    for res in final_results:
        d = res["dataset"]
        l = res["input_language"]
        
        if diff_stats.get(d) is None:
            diff_stats[d] = {}
        if diff_stats[d].get(l) is None:
            diff_stats[d][l] = {"total": 0, "correct": 0}

        diff_stats[d][l]["total"] += 1
        if res["is_correct_in_ground_truth"]:
            diff_stats[d][l]["correct"] += 1
        
    # 印出整體與分組準確率
    print("\n" + "="*50)
    print("📊 詳細效能分析 (Performance Breakdown)")
    print("="*50)
    
    summary_report = {} # 用來存入 JSON 的總結數據

    for d in diff_stats.keys():
        print(f"🔸 資料集: {d}")
        summary_report[d] = {}
        
        for l in diff_stats[d]:
            c = diff_stats[d][l]["correct"]
            t = diff_stats[d][l]["total"]
            acc = c / t if t > 0 else 0
            print(f"   - 語言 [{l:<8}]: diff 準確率 {acc:>6.2%} ({c}/{t})")

            # 🎯 防呆：使用 .get() 避免某個資料集剛好沒有 all-same 的題目而發生 KeyError
            same_c = same_status.get(d, {}).get("correct", 0)
            same_t = same_status.get(d, {}).get("total", 0)

            all_c = c + same_c * (1 - args.split)
            all_t = t + same_t * (1 - args.split)
            all_acc = all_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: all  準確率 {all_acc:>6.2%} ({all_c:.1f}/{all_t:.1f})")

            upper_bound_c = same_c * (1 - args.split) + t
            upper_bound_acc = upper_bound_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: upper bound all  準確率 {upper_bound_acc:>6.2%} ({upper_bound_c:.1f}/{all_t:.1f})")

            random_c = same_c * (1 - args.split) + t * 0.5
            random_acc = random_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: random all  準確率 {random_acc:>6.2%} ({random_c:.1f}/{all_t:.1f})")

            # 將這份數據儲存到 summary_report 字典中
            summary_report[d][l] = {
                "diff_correct": c,
                "diff_total": t,
                "diff_accuracy": acc,
                "all_correct": all_c,
                "all_total": all_t,
                "all_accuracy": all_acc,
                "random": random_acc,
                "upper_bound": upper_bound_acc
            }

        print("-" * 40)
            
    # 5. 結合 Summary 與 Predictions 輸出 JSON 檔案
    output_data = {
        "summary": summary_report,
        "predictions": final_results
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"💾 完整 JSON 報告 (含準確率 Summary) 已儲存至: {args.output_file}")

if __name__ == "__main__":
    main()