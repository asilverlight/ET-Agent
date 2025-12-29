import os
import sys

sys.path.append(os.getcwd())

import json
import langid
import string
import asyncio
import requests
import pdfplumber
import aiolimiter

from typing import Union, Dict, Optional, List

from io import BytesIO
from typing import Tuple
from bs4 import BeautifulSoup
from vllm import SamplingParams
from urllib.parse import urlencode
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from tqdm.asyncio import tqdm as async_tqdm
from concurrent.futures import ThreadPoolExecutor

from ..vllm_client_pool import VLLMClientPool
from .cache_manager import BaseCacheManager
from .search_tool_serper import SerperSearchTool


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.36",
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

try:
    import lxml  # type: ignore  # noqa: F401
    _HAS_LXML = True
except Exception:
    _HAS_LXML = False

_LXML_FALLBACK_WARNED = False


class SerperSearchToolSDS(SerperSearchTool):
    """SerperSearchTool with SDS (Summarization, Document-awareness, Step-by-step reasoning) capabilities."""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: Union[str, List[str]],
        max_results: int = 10,
        result_length: int = 1000,
        gl: str = None,
        hl: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file=None,
        url_cache_file=None,
        max_doc_len=3000, 
        summ_model_urls=None,
        summ_model_path=None,
        summ_model_name=None,
        max_sequence_length=20000,
        use_summarize=True,
        max_doc_length_without_summarize=1000,
    ):
        super().__init__(
            api_key=api_key,
            max_results=max_results,
            result_length=result_length,
            gl=gl,
            hl=hl,
            max_retries=max_retries,
            retry_delay=retry_delay,
            requests_per_second=requests_per_second,
            search_cache_file=search_cache_file,
        )
        self._url_fetch_limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second * 10, time_period=1.0
        )
        self.url_cache_manager = BaseCacheManager(url_cache_file)
       
        self.max_doc_len = max_doc_len
        self.max_sequence_length = max_sequence_length
        self.use_summarize = use_summarize
        self.max_doc_length_without_summarize = max_doc_length_without_summarize
        self.session = requests.Session()
        self.session.headers.update(headers)
        assert (
            summ_model_urls is not None
            and summ_model_name is not None
            and summ_model_path is not None
        )

        self.summ_vllm_pool = VLLMClientPool(
            endpoints=summ_model_urls, default_model=summ_model_name
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            n=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            summ_model_path, trust_remote_code=True
        )

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

    async def url_fetch_worker(self, task_queue, urls_to_fetch, results):
        while not task_queue.empty():
            try:
                idx = await task_queue.get()
                url = urls_to_fetch[idx]
                loop = asyncio.get_event_loop()
                async with self._url_fetch_limiter:
                    result = await loop.run_in_executor(
                        self._executor, 
                        lambda: self.extract_text_from_url(url),
                    )
                    results[idx] = result
            except Exception as e:
                results[idx] = "[Cannot fetch this url]"
            task_queue.task_done()

    async def fetch_urls(self, urls_to_fetch):
        urls_to_fetch_filtered = [
            u for u in urls_to_fetch if self.url_cache_manager.in_cache(u) is False
        ]
        urls_to_fetch = urls_to_fetch_filtered
        total_examples = len(urls_to_fetch)
        if total_examples == 0:
            return

        results = [None] * total_examples
        task_queue = asyncio.Queue()
        for i in range(total_examples):
            await task_queue.put(i)
        workers = []
        for _ in range(min(10, total_examples)):
            workers.append(
                asyncio.create_task(
                    self.url_fetch_worker(task_queue, urls_to_fetch, results)
                )
            )
        processed = 0
        while processed < total_examples:
            completed = sum(1 for r in results if r is not None)
            if completed > processed:
                processed = completed
            await asyncio.sleep(0.1)
        await task_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for url, result in zip(urls_to_fetch, results):
            if result != "[Cannot fetch this url]":
                await self.url_cache_manager.add_to_cache(url, result)

    def extract_text_from_url(self, url):
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                return self.extract_pdf_text(url)
            parser_to_use = "lxml" if _HAS_LXML else "html.parser"
            try:
                soup = BeautifulSoup(response.text, parser_to_use)
            except Exception:
                global _LXML_FALLBACK_WARNED
                if parser_to_use == "lxml" and _HAS_LXML and not _LXML_FALLBACK_WARNED:
                    _LXML_FALLBACK_WARNED = True
                soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text[:20000]
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"
        except requests.exceptions.ConnectionError:
            return "Error: Connection error occurred"
        except requests.exceptions.Timeout:
            return "Error: Request timed out after 20 seconds"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def get_formatted_documents(self, relevant_info):
        formatted_documents = ""
        for i, doc_info in enumerate(relevant_info):
            url = doc_info["url"]
            raw_context = self.url_cache_manager.hit_cache(url) or ""
            doc_info["snippet"] = (
                doc_info["snippet"].replace("<b>", "").replace("</b>", "")
            )
            if (
                raw_context.startswith("HTTP error occurred:")
                or raw_context.startswith("Error: Connection error occurred")
                or raw_context.startswith("Error: Request timed out after 20 seconds")
                or raw_context.startswith("Unexpected error:")
            ):
                context = "Web Page content cannot fetch"
            else:
                success, filtered_context = extract_snippet_with_context(
                    raw_context, doc_info["snippet"], context_chars=self.max_doc_len
                )
                if success:
                    context = filtered_context
                else:
                    context = raw_context[: self.max_doc_len * 2]

            doc_info["context"] = context
            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += (
                json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
            )
        return formatted_documents

    def get_formatted_documents_no_summary(self, relevant_info):
        formatted_documents = ""
        for i, doc_info in enumerate(relevant_info):
            url = doc_info["url"]
            raw_context = self.url_cache_manager.hit_cache(url) or ""
            doc_info["snippet"] = (
                doc_info["snippet"].replace("<b>", "").replace("</b>", "")
            )
            if (
                raw_context.startswith("HTTP error occurred:")
                or raw_context.startswith("Error: Connection error occurred")
                or raw_context.startswith("Error: Request timed out after 20 seconds")
                or raw_context.startswith("Unexpected error:")
            ):
                context = "Web Page content cannot fetch"
            else:
                success, filtered_context = extract_snippet_with_context(
                    raw_context, doc_info["snippet"], context_chars=self.max_doc_length_without_summarize
                )
                if success:
                    context = filtered_context
                else:
                    context = raw_context[: self.max_doc_length_without_summarize] + '...' if len(raw_context) > self.max_doc_length_without_summarize else raw_context

            doc_info["context"] = context
            doc_info.pop("url", None)
            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += (
                json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
            )
        return formatted_documents

    async def generate_webpage_to_reasonchain(
        self, prev_reasoning, search_query, document
    ):
        user_prompt = get_webpage_to_reasonchain_instruction(
            prev_reasoning, search_query, document
        )
        prompt = {"role": "user", "content": user_prompt}
        output = await self.webpage_analysis_single(prompt)
        summary = extract_answer(output)
        return summary

    async def postprocess_search_result(self, query, response, **kwargs):
        sample_stat = kwargs.get("sample_stat")
        if not sample_stat or not sample_stat.get("logs"):
             raise ValueError("postprocess_search_result requires 'sample_stat' with 'logs' in kwargs")

        truncated_prev_reasoning = self.get_truncated_prev_reasoning(
            sample_stat["logs"]
        )

        relevant_info = self.extract_relevant_info(response)
        urls_to_fetch = [it["url"] for it in relevant_info]
        await self.fetch_urls(urls_to_fetch)
        
        formatted_documents = ""
        # print(self.use_summarize)
        if self.use_summarize:
            formatted_documents = self.get_formatted_documents(relevant_info)
        else:
            formatted_documents = self.get_formatted_documents_no_summary(relevant_info)
        if len(formatted_documents) > self.max_sequence_length:
            formatted_documents = formatted_documents[:self.max_sequence_length] + '...'

        if self.use_summarize:
            summary = await self.generate_webpage_to_reasonchain(
                prev_reasoning=truncated_prev_reasoning,
                search_query=query,
                document=formatted_documents,
            )
        else: 
            summary = formatted_documents
        return summary

    async def webpage_analysis_single(self, prompt) -> str:
        in_context = self.tokenizer.apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True
        )
        result = await self.summ_vllm_pool.generate(
            in_context,
            self.sampling_params,
        )
        return result.choices[0].text

    def extract_pdf_text(self, url):
        try:
            response = self.session.get(url, timeout=20)
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            cleaned_text = " ".join(full_text.split()[:600])
            return cleaned_text
        except requests.exceptions.Timeout:
            return "Error: Request timed out after 20 seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    def extract_relevant_info(self, search_results):
        useful_info = []
        if search_results is None or "organic" not in search_results:
            return useful_info
        
        for i, result in enumerate(search_results["organic"][: self._max_results]):
            snippet = result.get("description") or result.get("snippet") or ""
            info = {
                "id": i + 1,
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "site_name": result.get("displayed_link", ""), # Serper uses 'displayed_link' for site name
                "date": result.get("date", "").split("T")[0], # Serper sometimes provides a 'date' field
                "snippet": snippet,
                "context": "",
            }
            useful_info.append(info)
        return useful_info


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 2500
) -> Tuple[bool, str]:
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence
        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[: context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_answer(output):
    extracted_text = ""
    if output is None:
        output = "None"
    pattern_info = "**Final Information**"
    pattern_step = "**Modified Reasoning Steps**"
    if pattern_info in output:
        extracted_text = (
            output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
        )
    elif pattern_step in output:
        extracted_text = output.split(pattern_step)[-1].strip("```").strip()
    else:
        extracted_text = output
    return extracted_text


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
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
