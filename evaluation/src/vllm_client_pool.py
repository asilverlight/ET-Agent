import sys
import os
sys.path.append(os.getcwd())

import asyncio
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI


class VLLMClientPool:
    
    def __init__(self, endpoints: List[str], api_keys: Optional[List[str]] = None, default_model: str = "Qwen2.5-72B-Instruct"):
        """
        
        Args:
            endpoints: ['http://...', 'http://...', ...]
            api_keys: list of api key for each endpoint
            default_model: default model name
        """
        self.clients = []
        api_keys = api_keys or ['EMPTY'] * len(endpoints)
        
        if len(api_keys) != len(endpoints):
            raise ValueError("len(api_keys) != len(endpoints)")
        for endpoint, api_key in zip(endpoints, api_keys):
            print(endpoint)
            self.clients.append(
                AsyncOpenAI(
                    base_url=endpoint,
                    api_key=api_key
                )
            )
        self.default_model = default_model
        self.current_client_idx = 0
        self.lock = asyncio.Lock()
        self.session_to_client = {}
        print(f"Initialized vllm clients pool with {len(endpoints)} endpoints")
    
    async def get_client_for_session(self, session_id: Optional[str] = None) -> AsyncOpenAI:
        async with self.lock:
            if not session_id:
                client = self.clients[self.current_client_idx]
                self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
                return client
            if session_id in self.session_to_client:
                return self.clients[self.session_to_client[session_id]]
            client_idx = self.current_client_idx
            self.session_to_client[session_id] = client_idx
            self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
            return self.clients[client_idx]
    
    async def generate(self, prompt: str, sampling_params: 'SamplingParams', session_id: Optional[str] = None) -> Any:
        params = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
            "stop": sampling_params.stop,
            "repetition_penalty": sampling_params.repetition_penalty,
            "include_stop_str_in_output": sampling_params.include_stop_str_in_output,
        }
        client = await self.get_client_for_session(session_id)
        max_attempts = 4
        for attempt in range(max_attempts): 
            try:
                response = await client.completions.create(
                    model=params.get("model", self.default_model),
                    prompt=prompt,
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.7),
                    max_tokens=params.get("max_tokens", 8192),
                    stop=params.get("stop", ["</python>", "</search>", "</answer>"]),
                    extra_body={
                        "repetition_penalty": params.get("repetition_penalty", 1.05),
                        "include_stop_str_in_output": params.get("include_stop_str_in_output", True),
                    }
                )
                return response
            except Exception as e:
                print(f"LLM request fails: {e}")
                if attempt == max_attempts - 1: 
                    return await self._retry_with_other_client(prompt, params, session_id)
        return None
    
    async def _retry_with_other_client(self, prompt: str, params: Dict[str, Any], session_id: Optional[str] = None) -> Any:
        """Retry using other clients"""
        original_client_idx = self.session_to_client.get(session_id, self.current_client_idx)
        tried_clients = set([original_client_idx])
        while len(tried_clients) < len(self.clients):
            async with self.lock:
                next_idx = (original_client_idx + 1) % len(self.clients)
                while next_idx in tried_clients:
                    next_idx = (next_idx + 1) % len(self.clients)
                tried_clients.add(next_idx)
                if session_id:
                    self.session_to_client[session_id] = next_idx
            client = self.clients[next_idx]
            try:
                response = await client.completions.create(
                    model=params.get("model", self.default_model),
                    prompt=prompt,
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.7),
                    max_tokens=params.get("max_tokens", 8192),
                    stop=params.get("stop", ["</python>", "</search>", "</answer>"]),
                    extra_body={
                        "repetition_penalty": params.get("repetition_penalty", 1.05),
                        "include_stop_str_in_output": params.get("include_stop_str_in_output", True),
                    }
                )
                return response
            except Exception as e:
                print(f"Client Retry failed: {str(e)}")
        print("All vllm clients fails, return None")
        return None 