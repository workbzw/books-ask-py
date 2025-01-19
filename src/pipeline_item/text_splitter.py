from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_step import Step

class TextSplitterStep(Step):
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    async def process(self, data: List[str]) -> List[str]:
        chunks = []
        for text in data:
            text_chunks = self.text_splitter.split_text(text)
            chunks.extend(text_chunks)
        return chunks