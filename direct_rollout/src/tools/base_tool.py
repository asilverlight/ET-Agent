import sys
import os

sys.path.append(os.getcwd())
from abc import ABC, abstractmethod


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def trigger_tag(self) -> str:
        """<xxx> </xxx>"""
        pass

    @abstractmethod
    async def execute(self, content: str, **kwargs) -> str:
        """Execute the tool."""
        pass
