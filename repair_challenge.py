from argparse import ArgumentParser
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from Strategy.RunContext import RunContext
from Strategy.StrategyConfig import StrategyConfig
from Strategy.RepairChallenge import RepairChallenge
from Log.NoLog import NoLog
from Log.TwoAgentLog import TwoAgentLog
from File.File import File

def parseArgs():
    parser = ArgumentParser(description="Multithreaded Repair Challenge Mode")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")
    
    # 接收目錄路徑
    parser.add_argument("--dirpath", required=True, help="Directory containing the Challenge JSON files to repair")
    
    # 控制執行緒數量的參數
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max concurrent threads/workers")
    
    args = parser.parse_args()
    return args

def runRepairTask(file_path, args):
    """
    單一檔案的修復任務，交由獨立的執行緒執行。
    """
    # Challenge 模式使用 TwoAgentLog
    log = TwoAgentLog() if args.log else NoLog()
    
    try:
        # 讀取目標檔案
        target_file = File(file_path)
    except Exception as e:
        print(f"❌ 無法讀取或解析檔案 {os.path.basename(file_path)}: {e}")
        return file_path, -1

    # 建立修復策略的 Config
    strategy_config_dict = {
        'strategyType': 'repairchallenge',
    }
    strategy_config = StrategyConfig.from_dict(strategy_config_dict)
    
    try:
        # 建立 Repair Challenge 策略
        strategy = RepairChallenge(strategy_config, log, target_file)

        # 放入執行環境並啟動
        context = RunContext()
        context.setStrategy(strategy)
        
        # 自動掃描錯誤、重新進行辯論與裁判，並覆寫回 JSON 檔
        repaired_ids = context.runExperiment()
        
        repaired_count = len(repaired_ids) if repaired_ids is not None else -1
        return file_path, repaired_count
        
    except Exception as e:
        print(f"❌ 修復檔案 {os.path.basename(file_path)} 時發生錯誤: {e}")
        return file_path, -1

def main():
    args = parseArgs()
    
    if not os.path.exists(args.dirpath):
        print(f"❌ 找不到目錄: {args.dirpath}")
        return

    # 尋找指定資料夾下所有的 .json 檔案
    search_pattern = os.path.join(args.dirpath, "*.json")
    files = glob.glob(search_pattern)

    if not files:
        print(f"⚠️ 在 {args.dirpath} 找不到任何 JSON 檔案。")
        return

    print(f"🚀 準備執行批次修復作業 (Run Challenge Batch Repair)...")
    print(f"📁 目標資料夾: {args.dirpath}")
    print(f"📄 找到檔案數: {len(files)} 個")
    print(f"⚙️ 執行緒數量: {args.workers}\n")

    # 啟動多執行緒處理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任務到執行緒池
        futures = {
            executor.submit(runRepairTask, file_path, args): file_path 
            for file_path in files
        }

        # 捕捉並回報結果
        for future in as_completed(futures):
            file_path = futures[future]
            file_name = os.path.basename(file_path)
            
            try:
                path, repaired_count = future.result()
                
                if repaired_count > 0:
                    print(f"🎉 完美修復！ {file_name} 共計重新辯論了 {repaired_count} 筆錯誤紀錄。")
                elif repaired_count == 0:
                    print(f"✨ 檢查完畢！ {file_name} 狀態完美，無需修復。")
                else:
                    print(f"⚠️ 異常警告！ {file_name} 修復程序未正常回傳結果。")
                    
            except Exception as e:
                print(f"❌ 處理 {file_name} 時發生無法預期的例外狀況: {e}")

    print("\n✅ 所有 Challenge 修復作業完成！你可以安心執行算分成式碼 (TestEM) 了。")

if __name__ == '__main__':
    main()