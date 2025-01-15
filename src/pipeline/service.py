import os
from src.pipeline.pipeline import (
    Pipeline,
    DataScrapingStep,
    DataCleaningStep,
    ChunkingStep,
    EmbeddingStep,
    PineconeSaveStep
)
from dotenv import load_dotenv
from .rag_pipeline import RAGPipeline

load_dotenv()

class PipelineService:
    def __init__(self):
        self.pipeline = Pipeline()
        self.rag_pipeline = None

    def pipeline_book2vec_run(self, url: str):
    
        # 添加处理步骤https://zh.wikipedia.org/wiki/%E5%A6%84%E6%83%B3  https://zh.wikipedia.org/zh-cn/%E5%8D%A1%E5%85%A7%E5%9F%BA%E6%BA%9D%E9%80%9A%E8%88%87%E4%BA%BA%E9%9A%9B%E9%97%9C%E4%BF%82
        self.pipeline.add_step(DataScrapingStep(urls=[url]))
        self.pipeline.add_step(DataCleaningStep())
        self.pipeline.add_step(ChunkingStep(chunk_size=600, chunk_overlap=120))

        # 添加embedding和存储步骤
        self.pipeline.add_step(EmbeddingStep(model_name='all-MiniLM-L6-v2'))
        self.pipeline.add_step(PineconeSaveStep(
            api_key=os.getenv("PINECONE_API_KEY"),
            metadata_index_url=url,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        ))
        
        # 运行pipeline
        result = self.pipeline.run()
        print("Pipeline result:", result)
        return result
    
    def init_rag_pipeline(self):
        """初始化RAG Pipeline"""
        if not self.rag_pipeline:
            self.rag_pipeline = RAGPipeline(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT"),
                index_name=os.getenv("PINECONE_INDEX_NAME")
            )
    
    def ask_book(self, question: str):
        """图书问答服务"""
        self.init_rag_pipeline()
        return self.rag_pipeline.query(question)

    async def ask_book_stream(self, question: str,source:str):
        """流式图书问答服务"""
        self.init_rag_pipeline()
        async for chunk in self.rag_pipeline.query_stream(question,source):
            yield chunk

if __name__ == "__main__":
    pipeline_service = PipelineService()
    pipeline_service.pipeline_book2vec_run("https://www.baidu.com")
