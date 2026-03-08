import sys
from argparse import ArgumentParser

from File.FileFactory import FileFactory
from DataSpliter.DataTransform import DataTransform

def parseArgs():
    parser = ArgumentParser(description="Transform Challenge Data to Training Data Format")
    
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the Challenge JSON files")
    parser.add_argument("-e", "--extension", default='*.json')
    # 改為接收 output_dir (輸出資料夾)
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for the combined training JSON files")
    
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    
    print(f"🚀 準備載入檔案 (Loading files from: {args.input_dir})...")
    
    # 1. 使用 FileFactory 批次讀取目標目錄下的所有 JSON 檔
    file_factory = FileFactory()
    files = file_factory.getFileInDir(args.input_dir, extension=args.extension)
    
    if not files:
        print(f"❌ 錯誤: 在目錄 {args.input_dir} 中找不到 {args.extension}。")
        sys.exit(1)
        
    print(f"📄 成功載入 {len(files)} 個檔案，準備進行資料轉換 (Transforming data)...")
    
    # 2. 將載入的 File 物件列表交給 DataTransform 進行處理
    transformer = DataTransform(files)
    
    # 3. 匯出結果到指定的資料夾 (traverse 所有 model/dataset pair)
    transformer.export_to_dir(args.output_dir)

if __name__ == '__main__':
    main()