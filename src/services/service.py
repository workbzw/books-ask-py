import os
from typing import List
from dotenv import load_dotenv
from ..utils.rag_utils import RAGUtils

load_dotenv()

class AskService:
    def __init__(self):
        self.rag_utils = None
    
    def init_rag_pipeline(self):
        """初始化RAG Pipeline"""
        if not self.rag_utils:
            self.rag_utils = RAGUtils(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT"),
                index_name=os.getenv("PINECONE_INDEX_NAME")
            )
    
    async def ask_book_stream(self, question: str,source:List[str]):
        """流式图书问答服务"""
        self.init_rag_pipeline()
        async for chunk in self.rag_utils.query_stream(question,source):
            yield chunk

if __name__ == "__main__":
    pipeline_service = AskService()
