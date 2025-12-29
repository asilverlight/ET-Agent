import sys
import os
sys.path.append(os.getcwd())
import requests
from typing import Optional, Union, Dict, List, Tuple
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
import aiolimiter
import langid
import string
import jieba
from typing import Tuple
import re
from tqdm.asyncio import tqdm as async_tqdm
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import pdfplumber
from io import BytesIO
from openai import OpenAI
from vllm import SamplingParams

from .cache_manager import BaseCacheManager
from .search_tool import BingSearchTool
from transformers import AutoTokenizer
from ..vllm_client_pool import VLLMClientPool

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

try:
    import lxml  # type: ignore  # noqa: F401
    _HAS_LXML = True
except Exception:
    _HAS_LXML = False

_LXML_FALLBACK_WARNED = False


class BingSearchToolSDScn(BingSearchTool):
    """BingSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: str,
        zone: str = "your_zone",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file=None,
        url_cache_file=None,
        max_doc_len=3000,  
        summ_model_urls=None,
        summ_model_path=None,
        summ_model_name=None,
    ):
        super().__init__(
            api_key=api_key,
            zone=zone,
            max_results=max_results,
            result_length=result_length,
            location=location,
            max_retries=max_retries,
            retry_delay=retry_delay,
            requests_per_second=requests_per_second,
            search_cache_file=search_cache_file,
        )
        self._url_fetch_limiter = aiolimiter.AsyncLimiter(max_rate=requests_per_second * 10, time_period=1.0)
        self.url_cache_manager = BaseCacheManager(url_cache_file)
       
        self.max_doc_len = max_doc_len
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

    def _pack_query(self, query):
        if langid.classify(query)[0] == 'zh':
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        input_obj = {
            "q": query, 
            "mkt": mkt, 
            "setLang": setLang, 
            "num": 11,
            "textDecorations": True, 
            "textFormat": "HTML" 
        }
        encoded_query = urlencode(input_obj)
        return encoded_query

    def get_truncated_prev_reasoning(self, reasoning_logs):
        assert len(reasoning_logs) > 0
        if type(reasoning_logs[0]) == dict:
            reasoning_logs = [message['content'] for message in reasoning_logs]
        prev_steps = [f"Step {i + 1}: {step}" for i, step in enumerate(reasoning_logs)]

        if len(prev_steps) <= 5:
            truncated_prev_reasoning = '\n\n'.join(prev_steps)
        else:
            truncated_prev_reasoning = ""
            for i, step in enumerate(prev_steps):
                if i == 0 \
                    or i >= len(prev_steps) - 4 \
                    or ('<search>' in step and '</search>' in step) \
                    or ('<result>' in step and '</result>' in step and '<search>' in prev_steps[i - 1]):
                    truncated_prev_reasoning += step + '\n\n'
                else:
                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                        truncated_prev_reasoning += '...\n\n'
        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
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
                print(f"url_fetch_worker error: {str(e)}")
                results[idx] = '[Cannot fetch this url]'
            task_queue.task_done()

    async def fetch_urls(self, urls_to_fetch):
       
        urls_to_fetch_filtered = [u for u in urls_to_fetch if self.url_cache_manager.in_cache(u) is False]
        print(f"{len(urls_to_fetch) - len(urls_to_fetch_filtered)} urls in caches, {len(urls_to_fetch_filtered)} urls to fetch")
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
            workers.append(asyncio.create_task(self.url_fetch_worker(task_queue, urls_to_fetch, results)))

       
        pbar = async_tqdm(total=total_examples, desc="Fetching urls...", ncols=70)
       
        processed = 0
        while processed < total_examples:
            completed = sum(1 for r in results if r is not None)
            if completed > processed:
                pbar.update(completed - processed)
                processed = completed
            await asyncio.sleep(0.1)
        pbar.close()
       
        await task_queue.join()
       
        for w in workers:
            w.cancel()
       
        await asyncio.gather(*workers, return_exceptions=True)
        for url, result in zip(urls_to_fetch, results):
            if result != '[Cannot fetch this url]':
                await self.url_cache_manager.add_to_cache(url, result)

    def extract_text_from_url(self, url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
        try:
            if use_jina:
                jina_headers = {
                    'Authorization': f'Bearer {jina_api_key}',
                    'X-Return-Format': 'markdown',
                   
                }
                response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
               
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
                print("use jina to extract text successfully")
            else:
                response = self.session.get(url, timeout=20) 
                response.raise_for_status() 
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type:
                    print("Extracting PDF...")
                    return self.extract_pdf_text(url)
               
                parser_to_use = 'lxml' if _HAS_LXML else 'html.parser'
                try:
                    soup = BeautifulSoup(response.text, parser_to_use)
                except Exception:
                    global _LXML_FALLBACK_WARNED
                    if parser_to_use == 'lxml' and _HAS_LXML and not _LXML_FALLBACK_WARNED:
                        print("lxml parser failed, using html parser")
                        _LXML_FALLBACK_WARNED = True
                    soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)

            if snippet:
                success, context = extract_snippet_with_context(text, snippet)
                if success:
                    print("Successfully extract page content based on snippet!")
                    return context
                else:
                    print(f"Failed to extrace page content based on snippet! Use the ogirin page: {context[:50]}...")
                    return text
            else:
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
            url = doc_info['url']
            raw_context = self.url_cache_manager.hit_cache(url) or ''
            doc_info['snippet'] = doc_info['snippet'].replace('<b>', '').replace('</b>', '')
            
            if raw_context.startswith('HTTP error occurred:') or \
            raw_context.startswith('Error: Connection error occurred') or \
            raw_context.startswith('Error: Request timed out after 20 seconds') or \
            raw_context.startswith('Unexpected error:'):
                print("Web page not accessible: ", raw_context[:50])
                context = 'Web Page content cannot fetch'
            else:
                success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=self.max_doc_len)
                if success:
                    print("Successfully extracted content from the web page based on the snippet!")
                    context = filtered_context
                else:
                    context = raw_context[:self.max_doc_len * 2]
                    print(f"Failed to extract content using the snippet. Falling back to raw page content, truncated length: {len(context)}, preview: {context[:50]}!")

            doc_info['context'] = context
            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        
        print('Total number of characters in formatted documents: ', len(formatted_documents))
        return formatted_documents

    async def generate_webpage_to_reasonchain(self, prev_reasoning, search_query, document):
        user_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document)
        prompt = {"role": "user", "content": user_prompt}
        print('Total number of characters in the prompt passed to summarization: ', len(user_prompt))
        output = await self.webpage_analysis_single(prompt)
        if output is None or output == "None" or output == "":
            print("summarization_error!!!")
        summary = extract_answer(output, mode='infogen')
        return summary


    async def postprocess_search_result(self, query, response, **kwargs):
        """
        Fetch web pages and generate a summary based on their content.

        Arguments:
            query {str} -- Search query.
            response {idk} -- Raw search results, typically more than 10 entries.
        """

       
        sample_stat = kwargs['sample_stat']
       
        truncated_prev_reasoning = self.get_truncated_prev_reasoning(sample_stat['logs'])

        relevant_info = self.extract_relevant_info(response)
        urls_to_fetch = [it['url'] for it in relevant_info]
        await self.fetch_urls(urls_to_fetch)
       
        formatted_documents = self.get_formatted_documents(relevant_info)
       
        summary = await self.generate_webpage_to_reasonchain(
            prev_reasoning=truncated_prev_reasoning,
            search_query=query,
            document=formatted_documents,
        )
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
            if response.status_code != 200:
                return f"Error: Unable to retrieve the PDF (status code {response.status_code})"

            with pdfplumber.open(BytesIO(response.content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
           
            cleaned_text = ' '.join(full_text.split()[:600])
            return cleaned_text
        except requests.exceptions.Timeout:
            return "Error: Request timed out after 20 seconds"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def extract_relevant_info(self, search_results):
        useful_info = []

        if search_results == None:
            return useful_info

        if 'organic' in search_results :
            for id, result in enumerate(search_results['organic'][:self._max_results]):
                info = {
                    'id': id + 1, 
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'site_name': result.get('site_name', ''),
                    'date': result.get('datePublished', '').split('T')[0],
                    'snippet': result.get('description', ''), 
                   
                    'context': '' 
                }
                useful_info.append(info)
        else:
            print("No organic results found.")
        print(f"len of useful_info: {len(useful_info)}")
        return useful_info

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    try:
        full_text = full_text[:50000]

        snippet_clean = snippet.lower()
        snippet_clean = remove_punctuation(snippet_clean)

        best_sentence = None
        best_f1 = 0.2

       
        if is_chinese(snippet_clean):
            sentences = re.split(r'[。！？]', full_text)
        else:
            sentences = sent_tokenize(full_text)

        snippet_words = set(tokenize_text(snippet_clean))

        for sentence in sentences:
            sentence_clean = sentence.lower()
            sentence_clean = remove_punctuation(sentence_clean)
            sentence_words = set(tokenize_text(sentence_clean))
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
            return False, full_text[:context_chars * 2]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Failed to extract snippet context due to {str(e)}"

def tokenize_text(text: str) -> list:
    lang = langid.classify(text)[0]
    if lang == 'zh':
        return list(jieba.cut(text))
    else:
        return text.split()


def is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_answer(output, mode='gen'):
    extracted_text = ''
    if output is None:
        output = "None"
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip() 
    elif mode == 'infogen':
        pattern_info = "**最终信息**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = output
    else:
       
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1] 
            if mode in ['choose', 'qa']:
               
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1] 
                extracted_text = extracted_text.strip("()")
    return extracted_text

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""***任务说明***：

你的任务是根据以下输入内容阅读并分析网页：**前序推理步骤**、**当前搜索查询** 和 **已搜索网页内容**。你的目标是从 **已搜索网页内容** 中提取与 **当前搜索查询** 相关且有用的信息，并将这些信息无缝整合进 **前序推理步骤** 中，以继续对原始问题的推理。

**操作指南：**

1. **分析已搜索网页内容：**
- 仔细审阅每一条搜索结果页面的内容。
- 识别出与 **当前搜索查询** 相关的事实信息，这些信息可以帮助原始问题的推理过程。

2. **提取相关信息：**
- 从已搜索网页中选择能够直接推进 **前序推理步骤** 的内容。确保提取的信息准确且相关。

3. **输出格式：**
- 使用如下格式展示当前搜索查询的有用信息：以 **最终信息** 开头，如下所示。
**最终信息**

[有用的信息]

- **输入内容：**

- **前序推理步骤：**
{prev_reasoning}

- **当前搜索查询：**
{search_query}

- **已搜索网页内容：**
{document}

现在你应该分析每一条网页内容，并基于当前搜索查询 "{search_query}" 和前序推理步骤找出有用信息。
"""

