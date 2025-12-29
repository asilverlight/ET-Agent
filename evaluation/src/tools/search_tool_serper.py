import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Union, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

# 复用已有的基础类与缓存管理器（与原 tools 目录一致）
from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager


class SerperSearchTool(BaseTool):
    """Serper (serper.dev) 搜索工具：仅使用 API Key，无需 zone"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: Union[str, List[str]],
        max_results: int = 10,
        result_length: int = 1000,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file: str = "",
    ):
        """
        Args:
            api_key: Serper API Key（必填）。可為單個字串或字串列表；若為列表，將在請求中依序嘗試。
            max_results: 返回条目上限（仅影响格式化输出）
            result_length: 单条摘要截断长度
            gl: 地域（例如 'us', 'cn'）。若不提供，将基于语言自动选择
            hl: 语言（例如 'en', 'zh'）。若不提供，将基于查询自动判断
            max_retries: 單個 api_key 的失敗重試次數
            retry_delay: 重试前休眠秒数
            requests_per_second: QPS 限流
            search_cache_file: 缓存文件路径
        """
        # 規範化 api_key 為列表
        if isinstance(api_key, str):
            self._api_keys: List[str] = [api_key]
        elif isinstance(api_key, (list, tuple)):
            # 僅保留字串並去除空白
            keys = [str(k).strip() for k in api_key if isinstance(k, str) and str(k).strip()]
            if not keys:
                raise ValueError("api_key 列表為空或不包含有效字串")
            self._api_keys = keys
        else:
            raise TypeError("api_key 必須是 str 或 List[str]")

        self._max_results = max_results
        self._result_length = result_length
        self._gl = gl
        self._hl = hl
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second, time_period=1.0
        )
        self.search_cache_manager = PreprocessCacheManager(search_cache_file)

    @property
    def name(self) -> str:
        return "search"

    @property
    def trigger_tag(self) -> str:
        return "websearch"

    def _infer_lang_geo(self, query: str) -> tuple[str, str]:
        """根据查询语言推断 hl、gl（如未显式提供）。"""
        if self._hl and self._gl:
            return self._hl, self._gl
        lang = langid.classify(query)[0]
        if lang == "zh":
            return (self._hl or "zh"), (self._gl or "cn")
        else:
            return (self._hl or "en"), (self._gl or "us")

    def _call_request(self, payload: Dict, timeout: int):
        """使用多個 api_key 順序嘗試請求。

        規則：
        - 對於當前 api_key，最多重試 self._max_retries 次。
        - 若當前 api_key 連續失敗 self._max_retries 次，切換到下一個 api_key，重新構建 headers 後繼續。
        - 若所有 api_key 都失敗，返回 None。
        """
        url = "https://google.serper.dev/search"
        for key_idx, key in enumerate(self._api_keys):
            headers = {
                "X-API-KEY": key,
                "Content-Type": "application/json",
            }
            error_cnt = 0
            while error_cnt < self._max_retries:
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                    response.raise_for_status()
                    print(
                        f"Successfully called Serper API with key index {key_idx}! Payload: {payload}"
                    )
                    return response.json()
                except requests.exceptions.Timeout:
                    error_cnt += 1
                    print(
                        f"key_idx: {key_idx}, error_cnt: {error_cnt}, Serper request timed out ({timeout}s). payload: {payload}"
                    )
                    time.sleep(self._retry_delay)
                except requests.exceptions.RequestException as e:
                    error_cnt += 1
                    print(
                        f"key_idx: {key_idx}, error_cnt: {error_cnt}, Serper request error: {e}. payload: {payload}"
                    )
                    time.sleep(self._retry_delay)
            # 當前 key 失敗，切換下一個 key
            print(
                f"API key index {key_idx} failed after {self._max_retries} retries. Switching to next key. payload: {payload}"
            )
        print(
            f"Serper query failed for all API keys after per-key retries. payload: {payload}"
        )
        return None

    def _make_request(self, query: str, timeout: int):
        """
        调用 Serper 搜索接口。
        """
        hl, gl = self._infer_lang_geo(query)
        payload = {
            "q": query,
            "hl": hl,
            "gl": gl,
            # 可选："num": 10, "page": 1
        }
        # 若返回的 organic 为空，则最多额外重试 2 次（共 3 次）
        empty_retry = 0
        max_empty_retries = 2
        while True:
            resp = self._call_request(payload, timeout)
            if not resp:
                return resp
            organic = (resp or {}).get("organic", [])
            if organic:
                return resp
            if empty_retry >= max_empty_retries:
                print(
                    f"Serper returned empty 'organic' after {empty_retry + 1} attempts, stop retrying. payload: {payload}"
                )
                return resp
            empty_retry += 1
            print(
                f"'organic' is empty, retrying {empty_retry}/{max_empty_retries} ... payload: {payload}"
            )
            time.sleep(self._retry_delay)

    async def postprocess_search_result(self, query, response, **kwargs) -> str | None:
        data = response or {}
        organic = data.get("organic", [])
        if not organic:
            return None

        chunk_content_list = []
        seen_snippets = set()
        for item in organic:
            # Serper 常见字段：title, link, snippet。个别返回可能有 description
            snippet = (
                item.get("description")
                or item.get("snippet")
                or ""
            ).strip()
            if snippet and snippet not in seen_snippets:
                chunk_content_list.append(snippet)
                seen_snippets.add(snippet)
        if not chunk_content_list:
            return None

        # 格式化输出
        formatted = []
        for idx, snippet in enumerate(chunk_content_list[: self._max_results], 1):
            snippet = snippet[: self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        return "\n".join(formatted)

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """
        执行 Serper 搜索（带缓存与限流）。
        变更：若命中缓存但缓存中的 organic 为空，则触发一次重新检索，并用新结果覆盖缓存。
        """
        cache_obj = self.search_cache_manager.hit_cache(query)
        from_cache = cache_obj is not None
        if from_cache:
            print("hit cache: ", query)
            response = cache_obj
            # 如果缓存中的 organic 为空，则重新请求一次并替换缓存
            organic = (response or {}).get("organic", [])
            if not organic:
                print("Cache organic empty, refetching: ", query)
                loop = asyncio.get_event_loop()
                async with self._limiter:
                    new_response = await loop.run_in_executor(
                        self._executor, lambda: self._make_request(query, timeout)
                    )
                if new_response is not None:
                    await self.search_cache_manager.add_to_cache(query, new_response)
                    response = new_response
        else:
            loop = asyncio.get_event_loop()
            async with self._limiter:
                response = await loop.run_in_executor(
                    self._executor, lambda: self._make_request(query, timeout)
                )
            if response is None:
                return f"Serper search failed: {query}"
            await self.search_cache_manager.add_to_cache(query, response)
        assert response is not None
        result = await self.postprocess_search_result(query, response, **kwargs)
        return result  # 若无结果，返回 None，与原 BingSearchTool 行为一致

if __name__ == "__main__":
    tool = SerperSearchTool(
        api_key=[
            "your_serper_api_key",
        ],
        max_results=10,
        result_length=1000,
        requests_per_second=2.0,
        search_cache_file="/path/to/search_cache.db",
    )
    print('====================== SerperSearchTool initialized ======================')
    result = tool._make_request("\"2024年信息学院同等学力人员硕士学位申请\" 北京考点", 60)
    import json
    print(json.dumps(result, indent=4, ensure_ascii=False))
