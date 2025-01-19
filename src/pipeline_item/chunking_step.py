from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base_step import Step

class ChunkingStep(Step):
    """文本分块步骤"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def process(self, data: List[str]) -> List[str]:
        chunks = []
        for text in data:
            text_chunks = self.text_splitter.split_text(text)
            chunks.extend(text_chunks)
        return chunks 