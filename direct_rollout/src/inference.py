import asyncio
import time
import json
import os

from vllm import SamplingParams
from transformers import AutoTokenizer
from typing import Dict, Any, Optional
from tqdm.asyncio import tqdm as async_tqdm

from .prompt_manager import PromptManager
from .data_loader import DataLoader
from .tools.tool_executor import ToolExecutor
from .tools import PythonTool, BingSearchTool, BingSearchToolSDS, SummarizeTool, LocalSearchTool
from .vllm_client_pool import VLLMClientPool
from .sample_processor import SampleProcessor
from .prompts import *

# 异步推理主类
class AsyncInference:

    def __init__(self, args):
        self.args = args
        self.use_log = args.use_log
        # 初始化VLLM推理池，用于管理多个推理服务端点
        self.vllm_pool = VLLMClientPool(
            endpoints=args.endpoints_infer,
            api_keys=args.api_keys,
            default_model=args.default_model_infer,
        )
        # 初始化提示词管理器
        self.prompt_manager = PromptManager(args.prompt_type)
        # 加载分词器
        self.tokenizer_infer = AutoTokenizer.from_pretrained(
            args.model_path_infer, trust_remote_code=True
        )
        # 先初始化summarize_tool
        self.summarize_tool = SummarizeTool(
            summ_model_urls=args.summ_model_urls,
            summ_model_path=args.summ_model_path,
            summ_model_name=args.summ_model_name,
            tokenizer=self.tokenizer_infer,
        ) if self.args.use_sds else None
        # 然后初始化工具执行器（如Python、搜索等工具）
        self.tool_executor = self._create_tool_executor()
        # 构建数据加载器列表，每个数据集一个DataLoader
        self.data_loader = DataLoader(self.args)
        # 构建采样参数（带停止词）
        self.args.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            n=1,
            include_stop_str_in_output=args.include_stop_str_in_output,
        )
        # 构建采样参数（不带停止词）
        self.args.sampling_params_nostop = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            n=1,
            include_stop_str_in_output=args.include_stop_str_in_output,
            stop=None,
        )
        # 单个样本的超时时间（秒），默认240秒
        self.sample_timeout = getattr(args, "sample_timeout", 240)
        print(f"Initialized {self.__class__.__name__}...")

    # 创建工具执行器，注册Python工具和Bing搜索工具
    def _create_tool_executor(self):
        tool_executor = ToolExecutor()
        # 注册Python代码执行工具
        python_tool = PythonTool(
            conda_env=self.args.conda_env,
            max_concurrent=self.args.python_max_concurrent,
        )
        tool_executor.register_tool(python_tool)
        # 注册Bing搜索工具
        if not self.args.compatible_search:
            if self.args.use_local_search:
                self.search_tool = LocalSearchTool(
                    local_search_url=self.args.local_search_url,
                    max_results=self.args.search_max_results,
                    result_length=self.args.search_result_length,
                    summarize_tool=self.summarize_tool,
                    use_summary=self.args.use_sds,
                )
                tool_executor.register_tool(self.search_tool)
            else:
                self.search_tool = BingSearchTool(
                    api_key=self.args.bing_api_key,
                    zone=self.args.bing_zone,
                    max_results=self.args.search_max_results,
                    result_length=self.args.search_result_length,
                    requests_per_second=self.args.bing_requests_per_second,
                    search_cache_file=self.args.search_cache_file,
                    max_retries=self.args.bing_max_retries,
                    retry_delay=self.args.bing_retry_delay,
                    use_log=self.args.use_log,
                )
                tool_executor.register_tool(self.search_tool)
        else:
            local_search_tool = LocalSearchTool(
                local_search_url=self.args.local_search_url,
                max_results=self.args.search_max_results,
                result_length=self.args.search_result_length,
                summarize_tool=self.summarize_tool,
                use_summary=self.args.use_sds,
            )
            tool_executor.register_tool(local_search_tool)
            web_search_tool = BingSearchTool(
                api_key=self.args.bing_api_key,
                zone=self.args.bing_zone,
                max_results=self.args.search_max_results,
                result_length=self.args.search_result_length,
                requests_per_second=self.args.bing_requests_per_second,
                search_cache_file=self.args.search_cache_file,
                max_retries=self.args.bing_max_retries,
                retry_delay=self.args.bing_retry_delay,
                use_log=self.args.use_log,
            )
            tool_executor.register_tool(web_search_tool)
        return tool_executor

    # 获取样本处理器（SampleProcessor），可被子类重载
    def get_processor(self, sample_stat, session_id):
        processor = SampleProcessor(
            self.prompt_manager,
            self.tool_executor,
            self.vllm_pool,
            self.tokenizer_infer,
            self.args,
            sample_stat,
            session_id,
        )
        return processor

    # 处理单个样本，返回推理结果
    async def process_sample(
        self, question: str, golden_answer: str, source: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # 构造样本状态字典
        sample_stat = {
            "instruction": ToolStar_Prompt,
            "input": question,
            "answer": golden_answer,
            "source": source,
            "rollout_outputs": [],
            "predictions": [],
            "tool_counts": [],
        }

        # 记录当前任务的中间结果，便于超时后获取
        current_task = asyncio.current_task()
        if current_task:
            setattr(current_task, "_current_result", sample_stat)

        # 获取样本处理器并执行推理
        processor = self.get_processor(sample_stat, session_id)
        await processor.run()
        # processor.log_timing()
        sample_stat = processor.sample_stat
        return sample_stat

    # 包装样本处理，增加超时和异常处理
    async def process_sample_wrap(self, idx, question, answer, source):
        try:
            # 创建异步任务
            process_task = asyncio.create_task(self.process_sample(question, answer, source))
            try:
                # 等待任务完成，超时则抛出TimeoutError
                result = await asyncio.wait_for(
                    process_task,
                    timeout=self.sample_timeout,
                )
            except asyncio.TimeoutError:
                # 超时处理，取消任务
                process_task.cancel()
                # 尝试获取部分中间结果
                partial_result = getattr(process_task, "_current_result", None)
                if partial_result:
                    partial_output = partial_result.get("initial_output", "Timeout")
                    partial_prediction = partial_result.get("prediction", "Timeout")
                    print(f"Sample timeout ({self.sample_timeout}): '{question}'")
                else:
                    partial_output = "Timeout"
                    partial_prediction = "Timeout"
                # 构造超时返回结果
                result = {
                    "instruction": ToolStar_Prompt,
                    "input": question,
                    "answer": answer,
                    "source": source,
                    "rollout_outputs": [],
                    "predictions": [],
                    "tool_counts": [],
                }
        except Exception as e:
            # 其他异常处理
            import traceback

            traceback.print_exc()
            print(f"Sample {idx} error: {str(e)}")
            result = {
                "instruction": ToolStar_Prompt,
                "input": question,
                "answer": answer,
                "source": source,
                "rollout_outputs": [],
                "predictions": [],
                "tool_counts": [],
            }
        return result

    # # 单轮推理接口（输入即问题，输出为推理结果）
    # async def run_inference(self, question):
    #     return await self.process_sample_wrap(question, question, None, None, None)

    # 异步任务worker，不断从队列取任务并处理
    async def task_worker(self, task_queue, questions, answers, sources, results):
        while not task_queue.empty():
            try:
                idx = await task_queue.get()
                
                
                # 执行单次采样
                results[idx] = await self.process_sample_wrap(
                    idx, questions[idx], answers[idx], sources[idx]
                )
                
                # 将采样结果添加到对应样本的结果列表中
                # results[idx].append(sample_result)
                
            except Exception as e:
                import traceback

                traceback.print_exc()
            task_queue.task_done()

    # 主推理流程，支持多数据集、多轮推理
    async def run(self):
        print(
            ">>> Inference dataset: ",
        )
        # 加载问题和答案
        self.source_datas = self.data_loader.load_data()
        # questions, answers, choices, formats = self.source_datas
        questions, answers, sources = [], [], []
        for i in range(len(self.source_datas)):
            questions.append(self.source_datas[i].get("question", self.source_datas[i].get("input", "")))
            answers.append(self.source_datas[i]["answer"])
            sources.append(self.source_datas[i]["source"])
        # 只取前counts个样本
        total_examples = min(len(questions), self.args.counts)
        print(
            f"Total examples: {total_examples}, Max concurrent requests: {self.args.max_concurrent_requests}"
        )
        results = [None] * total_examples  # 存储每个样本的推理结果
        start_time = time.time()
        task_queue = asyncio.Queue()  # 创建异步任务队列
        
        for i in range(total_examples):
            await task_queue.put(i)
        
        workers = []
        # 创建并发worker任务，数量不超过最大并发数和总任务数
        for _ in range(min(self.args.max_concurrent_requests, total_examples)):
            workers.append(
                asyncio.create_task(
                    self.task_worker(task_queue, questions, answers, sources, results)
                )
            )
        # 进度条显示
        pbar = async_tqdm(total=total_examples, desc="Processing samples")
        processed = 0
        while processed < total_examples:
            # 统计已完成的任务数
            completed = sum(1 for r in results if r is not None)
            if completed > processed:
                pbar.update(completed - processed)
                processed = completed
            await asyncio.sleep(0.1)
        pbar.close()
        await task_queue.join()  # 等待所有任务完成
        # 取消所有worker任务
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        end_time = time.time()
        print(f"Total Time: {(end_time - start_time) / 60:.2f}min")
        # 保存结果到json文件
        with open(self.args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(
            f"Processed {self.args.data_path}, save results to {self.args.output_path}"
        )
        print("Finished to process all datasets!")  # 所有数据集处理完成

# 继承AsyncInferenceCompletion，重载_create_tool_executor，使用SDS搜索工具
class AsyncInferenceCompletionSDS(AsyncInference):
    """使用SimpleDeepSearcher的Web浏览器工具"""

    def _create_tool_executor(self):
        tool_executor = ToolExecutor()
        # 注册Python工具
        python_tool = PythonTool(
            conda_env=self.args.conda_env,
            max_concurrent=self.args.python_max_concurrent,
        )
        tool_executor.register_tool(python_tool)
        if not self.args.compatible_search:
            if self.args.use_local_search:
                search_tool = LocalSearchTool(
                    local_search_url=self.args.local_search_url,
                    max_results=self.args.search_max_results,
                    result_length=self.args.search_result_length,
                    summarize_tool=self.summarize_tool,
                    use_summary=self.args.use_sds,
                )
                tool_executor.register_tool(search_tool)
            # 注册SDS搜索工具
            else:
                search_tool = BingSearchToolSDS(
                    api_key=self.args.bing_api_key,
                    zone=self.args.bing_zone,
                    max_results=self.args.search_max_results,
                    result_length=self.args.search_result_length,
                    requests_per_second=self.args.bing_requests_per_second,
                    max_retries=self.args.bing_max_retries,
                    retry_delay=self.args.bing_retry_delay,
                    search_cache_file=self.args.search_cache_file,
                    url_cache_file=self.args.url_cache_file,
                    summ_model_path=self.args.summ_model_path,
                    summ_model_urls=self.args.summ_model_urls,
                    summ_model_name=self.args.summ_model_name,
                    use_log=self.args.use_log,
                    max_sequence_length=self.args.max_search_result_length,
                )
                tool_executor.register_tool(search_tool)
        else:
            local_search_tool = LocalSearchTool(
                local_search_url=self.args.local_search_url,
                max_results=self.args.search_max_results,
                result_length=self.args.search_result_length,
                summarize_tool=self.summarize_tool,
                use_summary=self.args.use_sds,
            )
            tool_executor.register_tool(local_search_tool)
            web_search_tool = BingSearchToolSDS(
                    api_key=self.args.bing_api_key,
                    zone=self.args.bing_zone,
                    max_results=self.args.search_max_results,
                    result_length=self.args.search_result_length,
                    requests_per_second=self.args.bing_requests_per_second,
                    max_retries=self.args.bing_max_retries,
                    retry_delay=self.args.bing_retry_delay,
                    search_cache_file=self.args.search_cache_file,
                    url_cache_file=self.args.url_cache_file,
                    summ_model_path=self.args.summ_model_path,
                    summ_model_urls=self.args.summ_model_urls,
                    summ_model_name=self.args.summ_model_name,
                    max_sequence_length=self.args.max_search_result_length,
                )
            tool_executor.register_tool(web_search_tool)
        return tool_executor
