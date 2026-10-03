"""
check_results.py — 確認實驗結果完整性

用法:
    python3 scripts/legacy_eval/check_results.py result/baseline/
    python3 scripts/legacy_eval/check_results.py result/baseline/ --show-records   # 印出問題 records
    python3 scripts/legacy_eval/check_results.py result/baseline/ --threshold 5    # 只顯示缺失 > 5 筆的檔案
"""

import os
import json
import glob
import sys
from argparse import ArgumentParser


def check_dir(dir_path: str, show_records: bool = False, threshold: int = 0):
    if not os.path.exists(dir_path):
        print(f"❌ 找不到目錄: {dir_path}")
        return

    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    if not files:
        print(f"⚠️  {dir_path} 內找不到 JSON 檔案")
        return

    total_files = len(files)
    problem_files = []
    all_clean = True

    print(f"\n🔍 掃描目錄: {dir_path}  ({total_files} 個檔案)\n")
    print(f"{'檔案名稱':<60} {'缺失':>6} {'總數':>6} {'缺失率':>8}")
    print("-" * 85)

    for path in files:
        fname = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"{'⚠️  ' + fname:<60} {'JSON 格式錯誤':>20}")
            continue

        if not data or len(data) < 2:
            print(f"{'⚠️  ' + fname:<60} {'檔案為空':>20}")
            continue

        records = data[1:]
        missing = [
            r for r in records
            if not r.get("MyAnswer") or str(r.get("MyAnswer", "")).strip() in ("", "null", "None")
        ]
        total = len(records)
        miss_count = len(missing)
        rate = miss_count / total * 100 if total > 0 else 0.0

        if miss_count > threshold:
            all_clean = False
            flag = "⚠️ " if miss_count > 0 else "  "
            print(f"{flag + fname:<60} {miss_count:>6} {total:>6} {rate:>7.2f}%")
            problem_files.append((fname, missing, total))

            if show_records:
                for r in missing[:5]:  # 最多顯示 5 筆
                    q_id = r.get("id", "?")
                    result_preview = str(r.get("Result", "")).replace("\n", " ")[:120]
                    print(f"    id={q_id}  Result preview: {result_preview}")
                if len(missing) > 5:
                    print(f"    ... 還有 {len(missing) - 5} 筆")
                print()

    print("-" * 85)

    if all_clean:
        print(f"\n✅ 所有檔案的 MyAnswer 均完整（threshold={threshold}）")
    else:
        total_missing = sum(len(m) for _, m, _ in problem_files)
        total_records = sum(t for _, _, t in problem_files)
        print(f"\n📊 摘要：{len(problem_files)} 個檔案有缺失，共 {total_missing} 筆 / {total_records} 筆")


def main():
    parser = ArgumentParser(description="確認實驗結果 MyAnswer 完整性")
    parser.add_argument("dir", help="要掃描的結果資料夾")
    parser.add_argument("--show-records", action="store_true", help="印出有問題的 records")
    parser.add_argument("--threshold", type=int, default=0, help="只顯示缺失數 > N 的檔案（預設 0，全部顯示）")
    args = parser.parse_args()

    check_dir(args.dir, show_records=args.show_records, threshold=args.threshold)


if __name__ == "__main__":
    main()
