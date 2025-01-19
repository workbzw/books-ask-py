from typing import List
from .base_step import Step

class TextCleanerStep(Step):
    async def process(self, chunks: List[str]) -> List[str]:
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.strip()
            cleaned_chunk = ' '.join(cleaned_chunk.split())
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
        return cleaned_chunks 