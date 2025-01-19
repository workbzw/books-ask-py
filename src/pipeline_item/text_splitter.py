from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_step import Step

class TextSplitterStep(Step):
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process(self, text: str) -> List[str]:
        print("--------------text---------------")
        print(text)
        print("--------------text---------------")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_text(text) 