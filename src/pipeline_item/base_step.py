from abc import ABC, abstractmethod
from typing import Any

class Step(ABC):
    @abstractmethod
    async def process(self, data: Any) -> Any:
        pass 