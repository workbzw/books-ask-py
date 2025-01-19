from typing import Any, List
import requests
from bs4 import BeautifulSoup

from .base_step import Step


class DataScrapingStep(Step):
    """网页数据爬取步骤"""

    def __init__(self):
        pass

    async def process(self, data: Any = None) -> List[str]:
        texts = []
        urls:List[str] = data
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                # 这里可以根据具体网页结构调整提取方式
                text = soup.get_text(separator=' ', strip=True)
                texts.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
        return texts