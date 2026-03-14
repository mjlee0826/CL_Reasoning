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

    count_1, count_0 = 0, 0
    for y_list in val_y:
        if y_list == [1] * 10:
            count_1 += 1
        if y_list == [0] * 10:
            count_0 += 1
    print(count_1)
    print(count_0)
    
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
            
            # 🎯 移除 valid_probs_dict 的過濾邏輯，直接從 10 組中選出模型預測最高機率的組合
            best_pair = max(probs_dict, key=probs_dict.get)
                
            best_pair_idx = predictor.label_names.index(best_pair)
            is_correct = bool(y_list[best_pair_idx] == 1)
            
            # 🎯 精準計算 Random 期望值與 Upper Bound
            # 算出全部 10 組中，有幾組是正確的 (值為 1)
            correct_in_all_count = sum(y_list)
            
            # 隨機配對 (Random Pair)：從 10 組中隨機挑選 1 組的期望勝率
            expected_random_correct = correct_in_all_count / 10.0
            
            # 甲骨文上限 (Oracle Upper Bound)：10 組裡面只要「至少有一組」是對的，就視為正確 (1)
            is_upper_bound_correct = 1 if correct_in_all_count > 0 else 0
            
            final_results.append({
                "dataset": meta["dataset"],
                "q_id": meta["q_id"],
                "input_language": lang,
                "question": question_text,
                "best_predicted_pair": best_pair,
                "is_correct_in_ground_truth": is_correct,
                "expected_random_correct": expected_random_correct, # 存入單題期望值
                "is_upper_bound_correct": is_upper_bound_correct,   # 存入單題理論上限
                "predicted_probabilities": probs_dict,
                "actual_ground_truth_y": y_list 
            })

    # 4. 🎯 進行詳細的 Performance 成效分析
    diff_stats = {}
    
    for res in final_results:
        d = res["dataset"]
        l = res["input_language"]
        
        if diff_stats.get(d) is None:
            diff_stats[d] = {}
        if diff_stats[d].get(l) is None:
            diff_stats[d][l] = {
                "total": 0, 
                "correct": 0,
                "random_correct": 0.0,
                "upper_bound_correct": 0
            }

        diff_stats[d][l]["total"] += 1
        if res["is_correct_in_ground_truth"]:
            diff_stats[d][l]["correct"] += 1
            
        # 累加 Random 與 Upper Bound 的期望答對題數
        diff_stats[d][l]["random_correct"] += res["expected_random_correct"]
        diff_stats[d][l]["upper_bound_correct"] += res["is_upper_bound_correct"]
        
    print("\n" + "="*70)
    print("📊 詳細效能分析 (Performance Breakdown)")
    print("="*70)
    
    summary_report = {} 

    for d in diff_stats.keys():
        print(f"🔸 資料集: {d}")
        summary_report[d] = {}
        
        for l in diff_stats[d]:
            c = diff_stats[d][l]["correct"]
            t = diff_stats[d][l]["total"]
            diff_random_c = diff_stats[d][l]["random_correct"]
            diff_upper_c = diff_stats[d][l]["upper_bound_correct"]
            
            acc = c / t if t > 0 else 0
            print(f"   - 語言 [{l:<8}]: diff 準確率 {acc:>6.2%} ({c}/{t})")

            print(f"   - 語言 [{l:<8}]: upper bound 準確率 {acc:>6.2%} ({diff_upper_c}/{t})")

            print(f"   - 語言 [{l:<8}]: random baseline 準確率 {acc:>6.2%} ({diff_random_c}/{t})")

            # 抓取無分歧資料 (All Same)
            same_c = same_status.get(d, {}).get("correct", 0)
            same_t = same_status.get(d, {}).get("total", 0)

            # 🎯 轉換為精確的 Validation 整數題數，避免浮點數誤差
            val_same_t = same_t - int(same_t * args.split)
            val_same_c = same_c - int(same_c * args.split)

            # --- 全局 (All) 統計 ---
            all_c = c + val_same_c
            all_t = t + val_same_t
            all_acc = all_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: all  準確率 {all_acc:>6.2%} ({all_c:.1f}/{all_t:.1f})")

            # --- 精確計算 Upper Bound ---
            # all-same 的答對題數 + diff 中理論能答對的題數
            upper_bound_c = val_same_c + diff_upper_c
            upper_bound_acc = upper_bound_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: upper bound 準確率 {upper_bound_acc:>6.2%} ({upper_bound_c:.1f}/{all_t:.1f})")

            # --- 精確計算 Random Baseline ---
            # all-same 的答對題數 + diff 中根據 10 組選項分佈算出的隨機期望值
            random_c = val_same_c + diff_random_c
            random_acc = random_c / all_t if all_t > 0 else 0
            print(f"   - 語言 [{l:<8}]: random baseline 準確率 {random_acc:>6.2%} ({random_c:.1f}/{all_t:.1f})")

            summary_report[d][l] = {
                "diff_correct": c,
                "diff_total": t,
                "diff_accuracy": acc,
                "all_correct": all_c,
                "all_total": all_t,
                "all_accuracy": all_acc,
                "random_accuracy": random_acc,
                "upper_bound_accuracy": upper_bound_acc
            }

        print("-" * 60)
            
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