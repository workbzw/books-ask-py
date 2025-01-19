from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base_step import Step

class EmbeddingStep(Step):
    """文本嵌入步骤"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    async def process(self, data: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(data)
        return embeddings 