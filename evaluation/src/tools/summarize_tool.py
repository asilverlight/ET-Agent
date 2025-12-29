import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Union, Dict
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager
from vllm import SamplingParams
from transformers import AutoTokenizer
from typing import List

from ..vllm_client_pool import VLLMClientPool

def get_summarize_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

**Inputs:**
- **Previous Reasoning Steps:**
{prev_reasoning}

- **Current Search Query:**
{search_query}

- **Searched Web Pages:**
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


class SummarizeTool():

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(self, summ_model_urls: List[str], summ_model_path: str, summ_model_name: str, tokenizer: AutoTokenizer):
        self.clients = VLLMClientPool(summ_model_urls, default_model=summ_model_name)
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=4096,
            top_p=0.95,
            n=1,
            repetition_penalty=1.15,
        )

    async def summarize(self, query: str, **kwargs):
        prev_reasoning = kwargs["reasoning_path"]
        search_query = query
        document = kwargs["document"]
        user_prompt = get_summarize_instruction(prev_reasoning, search_query, document)
        prompt = {"role": "user", "content": user_prompt}
        if 'qwen3' in self.clients.default_model.lower():
            in_context = self.tokenizer.apply_chat_template(
                [prompt], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            in_context = self.tokenizer.apply_chat_template(
                [prompt], tokenize=False, add_generation_prompt=True
            )
        result = await self.clients.generate(
            in_context,
            self.sampling_params,
        )
        return result.choices[0].text