from typing import List, Tuple, Any
import numpy as np
from pinecone import Pinecone
import hashlib

from src.pipeline_item.base_step import Step

class PineconeSaveStep(Step):
    """保存到Pinecone步骤"""

    def __init__(self, api_key: str, source: str, index_name: str):
        self.api_key = api_key
        self.source = source
        self.index_name = index_name

    def _get_safe_filename_hash(self, filename: Any) -> str:
        """安全地获取文件名的哈希值"""
        # 确保文件名是字符串
        safe_filename = str(filename) if filename is not None else "unknown"
        return hashlib.md5(safe_filename.encode('utf-8')).hexdigest()[:8]

    async def process(self, data: Any) -> dict:
        print("--------------?data")
        print(data)
        print("--------------?data")
        try:
            # 如果 data 是元组，解构它
            if isinstance(data, tuple) and len(data) == 2:
                chunks, embeddings = data
            else:
                raise ValueError(f"Invalid data format received: {data}")
            
            # 初始化 Pinecone
            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(self.index_name)
            
            # 生成文件名的哈希值作为前缀
            filename_hash = self._get_safe_filename_hash(self.source)
            
            # 创建向量列表
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{filename_hash}_{i}"
                vectors.append((
                    vector_id,
                    embedding if isinstance(embedding, list) else embedding.tolist(),
                    {
                        "source": self.source,
                        "text": str(chunk) if chunk is not None else ""
                    }
                ))
            
            # 上传到 Pinecone
            if vectors:
                index.upsert(vectors=vectors)
            
            # 返回结果
            return {
                "filename": self.source,
                "status": "success",
                "chunks_count": len(vectors)
            }
            
        except Exception as e:
            print(f"Error in PineconeSaveStep: {str(e)}")
            raise 