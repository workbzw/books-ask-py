import os
import tempfile
from fastapi import UploadFile
from .base_step import Step

class TempFileSaverStep(Step):
    def __init__(self):
        self.temp_file_path = None

    async def process(self, file: UploadFile):
        """
        保存上传的文件到临时文件
        """
        try:
            # 创建临时文件，保持原始扩展名
            file_extension = os.path.splitext(file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                # 读取上传的文件内容并写入临时文件
                content = await file.read()
                # 确保文件指针被重置
                await file.seek(0)
                
                # 写入临时文件
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                self.temp_file_path = temp_file.name
                
            # 验证文件是否成功写入
            if not os.path.exists(self.temp_file_path):
                raise Exception("Failed to create temporary file")
                
            print(f"Temporary file created at: {self.temp_file_path}")
            return self.temp_file_path

        except Exception as e:
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)
            raise Exception(f"Error saving temporary file: {str(e)}")

    def cleanup(self):
        """
        清理临时文件
        """
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
                print(f"Temporary file cleaned up: {self.temp_file_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")