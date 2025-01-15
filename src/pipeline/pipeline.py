from abc import ABC, abstractmethod
from typing import Any, List
import requests
from bs4 import BeautifulSoup
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


class PipelineStep(ABC):
    """Pipeline步骤的抽象基类"""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据的抽象方法"""
        pass


class DataScrapingStep(PipelineStep):
    """网页数据爬取步骤"""

    def __init__(self, urls: List[str]):
        self.urls = urls

    def process(self, data: Any = None) -> List[str]:
        texts = []
        for url in self.urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                # 这里可以根据具体网页结构调整提取方式
                text = soup.get_text(separator=' ', strip=True)
                texts.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
        return texts


class DataCleaningStep(PipelineStep):
    """数据清洗步骤"""

    def process(self, data: List[str]) -> List[str]:
        cleaned_texts = []
        for text in data:
            # 实现你的数据清洗逻辑
            cleaned_text = text.strip()
            cleaned_text = ' '.join(cleaned_text.split())  # 删除多余空白
            cleaned_texts.append(cleaned_text)
        return cleaned_texts


class ChunkingStep(PipelineStep):
    """文本分块步骤"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, data: List[str]) -> List[str]:
        chunks = []
        for text in data:
            text_chunks = self.text_splitter.split_text(text)
            chunks.extend(text_chunks)
        return chunks


class EmbeddingStep(PipelineStep):
    """文本嵌入步骤"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def process(self, data: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(data)
        return embeddings


class PineconeSaveStep(PipelineStep):
    """保存到Pinecone步骤"""

    def __init__(self, api_key: str, metadata_index_url: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.metadata_index_url = metadata_index_url

    def process(self, data: tuple[List[str], List[np.ndarray]]) -> None:
        texts, embeddings = data
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append((
                str(i),
                embedding.tolist(),
                {"source": self.metadata_index_url, "text": text}
            ))
        self.index.upsert(vectors=vectors)
        return None


class Pipeline:
    """数据处理Pipeline类"""

    def __init__(self):
        self.result = None
        self.steps = []
        self.texts = None  # 存储原始文本
        self.embeddings = None  # 存储嵌入向量

    def add_step(self, step: PipelineStep) -> None:
        """添加处理步骤"""
        self.steps.append(step)

    def run(self, initial_data: Any = None) -> Any:
        """运行整个pipeline"""
        data = initial_data
        for step in self.steps:
            if isinstance(step, EmbeddingStep):
                # 在生成嵌入向量之前保存texts
                self.texts = data
                # 生成嵌入向量
                self.embeddings = step.process(data)
                data = self.embeddings
            elif isinstance(step, PineconeSaveStep):
                # 传递texts和embeddings给PineconeSaveStep
                step.process((self.texts, self.embeddings))
            else:
                data = step.process(data)

        self.result = data
        return {
            "texts": self.texts,
            "embeddings": self.embeddings,
            "result": self.result
        }

    def get_result(self):
        return self.result
