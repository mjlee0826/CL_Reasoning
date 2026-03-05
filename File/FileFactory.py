import os
import glob
from File.File import File  # 確保這裡引入了你封裝好的 File (或 ResultFile) 類別

class FileFactory():
    """
    A factory class responsible for instantiating File objects.
    Provides methods to load single files or batch load all files from a directory.
    """
    def __init__(self):
        pass
    
    def getFileByPath(self, path: str) -> File:
        """
        Instantiates and returns a single File object given its file path.
        """
        return File(path)
    
    def getFileInDir(self, dir_path: str, extension: str = "*.json") -> list[File]:
        """
        Scans a directory for files matching a specific extension and returns a list of File objects.
        
        Args:
            dir_path (str): The target directory to scan.
            extension (str): The file extension to look for. Defaults to "*.json".
            
        Returns:
            list[File]: A list containing successfully instantiated File objects.
        """
        # 1. 檢查資料夾是否存在
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"[FileFactory] Warning: Directory '{dir_path}' does not exist or is not a directory.")
            return []

        files_list = []
        
        # 2. 組合搜尋路徑 (例如: result/v2_tempature0/*.json)
        search_pattern = os.path.join(dir_path, extension)
        file_paths = glob.glob(search_pattern)
        
        if not file_paths:
            print(f"[FileFactory] Warning: No {extension} files found in '{dir_path}'.")
            return files_list

        # 3. 逐一實例化 File 物件
        for path in file_paths:
            try:
                # 嘗試建立 File 物件
                file_obj = File(path)
                files_list.append(file_obj)
            except Exception as e:
                # 若發生 JSONDecodeError 或 ValueError (檔案空白等)，則略過該檔案並印出警告
                print(f"[FileFactory] Error loading file {path}: {e}")

        # 依照檔名排序一下，讓回傳的列表順序比較可預期
        files_list.sort(key=lambda x: x.file_path)
        
        return files_list