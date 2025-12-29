import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Union, Dict, List
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager
from .summarize_tool import SummarizeTool

class LocalSearchTool(BaseTool):
    """BingSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        local_search_url: str,
        max_results: int = 10,
        summarize_tool: SummarizeTool = None,
    ):
        self._local_search_url = local_search_url
        self._max_results = max_results
        self._summarize_tool = summarize_tool

    @property
    def name(self) -> str:
        return "localsearch"

    @property
    def trigger_tag(self) -> str:
        return "search"

    async def postprocess_search_result(self, query, response, **kwargs) -> str:
        # reasoning_path = kwargs["sample_stat"]["logs"]
        reasoning_path = kwargs["existing_logs"]
        search_query = query
        document = response
        result = await self._summarize_tool.summarize(search_query, reasoning_path=self.get_truncated_prev_reasoning(reasoning_path), document=document)
        return result

    async def search(self, query: str) -> str:
        if query == '':
            return 'invalid query'
        url = f'http://{self._local_search_url}/search'
        data = {'query': query, 'top_n': self._max_results}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Local search failed: {query}"
    
    async def batch_search(self, queries: List[str]) -> List[str]:
        if len(queries) == 0:
            return ['invalid query']
        url = f'http://{self._local_search_url}/batch_search'
        data = {'queries': queries, 'top_n': self._max_results}
        response = requests.post(url, json=data)
        
        result_list = []
        for item in response.json():
            curr_result = ''
            for line in item:
                curr_result += f"{line['contents']}\n\n"
            result_list.append(curr_result.strip())
        return result_list

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """
        Execute a Bing search query with support for cache and semantic similarity cache hits.

        Args:
            query: The search query text.
            timeout: Request timeout in seconds.
            model: SBERT model used for semantic search.
            threshold: Minimum similarity threshold.
            top_k: Number of top most similar cached entries to consider.

        Returns:
            A string containing the search result.
        """
        response = await self.search(query)
        assert response is not None
        return await self.postprocess_search_result(query, response, **kwargs)

    def get_truncated_prev_reasoning(self, reasoning_logs):
        assert len(reasoning_logs) > 0
        if type(reasoning_logs[0]) == dict:
            reasoning_logs = [message["content"] for message in reasoning_logs]
        prev_steps = [f"Step {i + 1}: {step}" for i, step in enumerate(reasoning_logs)]

        if len(prev_steps) <= 5:
            truncated_prev_reasoning = "\n\n".join(prev_steps)
        else:
            truncated_prev_reasoning = ""
            for i, step in enumerate(prev_steps):
                if (
                    i == 0
                    or i >= len(prev_steps) - 4
                    or ("<search>" in step and "</search>" in step)
                    or (
                        "<result>" in step
                        and "</result>" in step
                        and "<search>" in prev_steps[i - 1]
                    )
                ):
                    truncated_prev_reasoning += step + "\n\n"
                else:
                    if truncated_prev_reasoning[-len("\n\n...\n\n") :] != "\n\n...\n\n":
                        truncated_prev_reasoning += "...\n\n"
        truncated_prev_reasoning = truncated_prev_reasoning.strip("\n")
        return truncated_prev_reasoning