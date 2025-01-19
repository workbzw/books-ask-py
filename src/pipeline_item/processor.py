from typing import List, Any
from .base_step import Step

class Processor:
    def __init__(self):
        self.steps: List[Step] = []

    def add_step(self, step: Step) -> 'Processor':
        self.steps.append(step)
        return self

    async def process(self, input_data: Any) -> Any:
        result = input_data
        for step in self.steps:
            result = await step.process(result)
        return result 