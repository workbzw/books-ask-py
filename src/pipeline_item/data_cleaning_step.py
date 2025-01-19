from typing import List
from .base_step import Step

class DataCleaningStep(Step):
    """数据清洗步骤"""

    async def process(self, data: List[str]) -> List[str]:
        cleaned_texts = []
        for text in data:
            # 实现你的数据清洗逻辑
            cleaned_text = text.strip()
            cleaned_text = ' '.join(cleaned_text.split())  # 删除多余空白
            cleaned_texts.append(cleaned_text)
        return cleaned_texts 