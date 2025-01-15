import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from src.pipeline.pipeline import (
    Pipeline,
    DataScrapingStep,
    DataCleaningStep,
    ChunkingStep,
    EmbeddingStep,
    PineconeSaveStep
)
from dotenv import load_dotenv

load_dotenv()

def test_pipeline():
    pipeline = Pipeline()
    
    # 添加处理步骤
    pipeline.add_step(DataScrapingStep(urls=["https://zh.wikipedia.org/zh-cn/%E5%8D%A1%E5%85%A7%E5%9F%BA%E6%BA%9D%E9%80%9A%E8%88%87%E4%BA%BA%E9%9A%9B%E9%97%9C%E4%BF%82"]))
    pipeline.add_step(DataCleaningStep())
    pipeline.add_step(ChunkingStep(chunk_size=600, chunk_overlap=120))

    # 添加embedding和存储步骤
    pipeline.add_step(EmbeddingStep(model_name='all-MiniLM-L6-v2'))
    pipeline.add_step(PineconeSaveStep(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        index_name=os.getenv("PINECONE_INDEX_NAME")
    ))
    
    # 运行pipeline
    result = pipeline.run()
    print("Pipeline result:", result)

if __name__ == "__main__":
    test_pipeline() 