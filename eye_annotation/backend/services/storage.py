import os
import uuid
from pathlib import Path
from typing import List, Union
from backend.config import settings

class StorageService:
    def __init__(self):
        self.storage_root = Path(settings.STORAGE_ROOT).resolve()
        self.categories = ['uploads', 'media', 'frames', 'masks', 'exports']
        self._init_directories()
        
    def _init_directories(self):
        self.storage_root.mkdir(parents=True, exist_ok=True)
        for category in self.categories:
            (self.storage_root / category).mkdir(exist_ok=True)
            
    def save_file(self, *args, **kwargs) -> str:
        """
        Flexible file saver that accepts either:
        - save_file(file_bytes, category, filename)
        - save_file(filename, file_bytes, folder="uploads")
        - keyword args: file_bytes, category/folder, filename
        """
        file_bytes = None
        category = "uploads"
        filename = "file.bin"

        if len(args) == 3:
            if isinstance(args[0], (bytes, bytearray)):
                file_bytes = bytes(args[0])
                category = str(args[1])
                filename = str(args[2])
            else:
                filename = str(args[0])
                file_bytes = bytes(args[1])
                category = str(args[2])
        elif len(args) == 2:
            if isinstance(args[0], (bytes, bytearray)):
                file_bytes = bytes(args[0])
                filename = str(args[1])
            else:
                filename = str(args[0])
                file_bytes = bytes(args[1])
        elif len(args) == 1:
            if isinstance(args[0], (bytes, bytearray)):
                file_bytes = bytes(args[0])
            else:
                filename = str(args[0])

        if "file_bytes" in kwargs:
            file_bytes = bytes(kwargs["file_bytes"])
        if "data" in kwargs:
            file_bytes = bytes(kwargs["data"])
        if "category" in kwargs:
            category = kwargs["category"]
        elif "folder" in kwargs:
            category = kwargs["folder"]
        if "filename" in kwargs:
            filename = kwargs["filename"]

        if file_bytes is None:
            raise ValueError("No file bytes provided to save_file")

        if category not in self.categories:
            self.categories.append(category)
            (self.storage_root / category).mkdir(parents=True, exist_ok=True)

        clean_filename = Path(filename).name
        unique_filename = f"{uuid.uuid4().hex[:8]}_{clean_filename}"
        relative_path = os.path.join(category, unique_filename)
        absolute_path = self.storage_root / relative_path
        
        with open(absolute_path, "wb") as f:
            f.write(file_bytes)
            
        return str(Path(relative_path).as_posix())
        
    def get_file_path(self, relative_path: str) -> Path:
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.storage_root / p

    def get_local_path(self, relative_path: str) -> str:
        return str(self.get_file_path(relative_path).resolve())
        
    def get_file_bytes(self, relative_path: str) -> bytes:
        absolute_path = self.get_file_path(relative_path)
        with open(absolute_path, "rb") as f:
            return f.read()
            
    def delete_file(self, relative_path: str) -> bool:
        absolute_path = self.get_file_path(relative_path)
        if absolute_path.exists() and absolute_path.is_file():
            absolute_path.unlink()
            return True
        return False
        
    def list_files(self, category: str) -> List[str]:
        category_dir = self.storage_root / category
        if not category_dir.exists():
            return []
        return [f.name for f in category_dir.iterdir() if f.is_file()]
