from typing import List
from sentence_transformers import SentenceTransformer
from .base_step import Step

class TextEmbeddingStep(Step):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    async def process(self, chunks: List[str]) -> List[List[float]]:
        return self.model.encode(chunks).tolist() 