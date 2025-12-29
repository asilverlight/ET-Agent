from args import get_args
import json
import pandas as pd
from tqdm import tqdm
import asyncio
import time
import random
import os
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer
import numpy as np
import math
import re

from prompts import *
from vllm_client_pool import *
from utils import *

class Evaluation:
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(42)

    def load_data(self, data_path=None, count=None):
        if data_path is None:
            data_path = self.args.data_path
        if data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        elif data_path.endswith(".parquet"):
            data = pd.read_parquet(data_path, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        temp_count = len(data)
        if self.args.count is not None:
            temp_count = self.args.count
        if count is not None:
            temp_count = count
        data = data[:temp_count]
        return data

    async def task_worker(self, task_queue: asyncio.Queue, results_2d, vllm_pool: VLLMClientPool):
        while True:
            try:
                info = await task_queue.get()
            except asyncio.CancelledError:
                break
            try:
                resp = await vllm_pool.generate(info['prompt'], session_id=info['session_id'])
                if resp is None:
                    text = "<answer>NA</answer>"
                else:
                    # OpenAI completions format
                    try:
                        text = (resp.choices[0].text or "").strip()
                    except Exception:
                        text = str(resp)
                results_2d[info['sample_idx']][info['resp_idx']] = text
            except Exception as e:
                results_2d[info['sample_idx']][info['resp_idx']] = "NA"
            finally:
                task_queue.task_done()

    async def _progress_monitor(self, results_2d, total_examples: int):
        # 每当某个样本的所有 round 都完成时，进度条 +1
        pbar = async_tqdm(total=total_examples, desc="Processing samples")
        processed = 0
        try:
            while processed < total_examples:
                completed = 0
                for row in results_2d:
                    if row is None:
                        continue
                    # 该样本所有轮次均完成
                    if all(x is not None for x in row):
                        completed += 1
                if completed > processed:
                    pbar.update(completed - processed)
                    processed = completed
                await asyncio.sleep(0.1)
        finally:
            pbar.close()

    @staticmethod
    def _extract_answer_text(text: str) -> str:
        if text is None:
            return ""
        # 提取 <answer> ... </answer> 中的内容
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _to_yes_no(token: str) -> str:
        if token is None:
            return ""
        s = token.strip().lower()
        # 清理常见标点
        s = s.strip("\t\n\r .,!?:;\"'()[]{}")
        # 仅看前缀，支持诸如 "yes, ..."/"no." 等
        if s.startswith("yes"):
            return "yes"
        if s.startswith("no"):
            return "no"
        return s

    @staticmethod
    def check_my_result_correctness(results):
        # INSERT_YOUR_CODE
        """
        如果任何一个result中，有”Execution Timeout“字样，或者”Tool execute failed“字样，
        或者"error" in result.lower()，或者result为空，都认为这个results是错误的
        """
        situations = []
        for result in results:
            if (
                result == "" or
                "Execution Timeout" in result or
                "Tool execute failed" in result or
                "error" in result.lower()
            ):
                situations.append(False)
            else:
                situations.append(True)
        return situations

    async def evaluate_redundancy(self):
        source_datas = self.load_data()
        vllmclientpool = VLLMClientPool(self.args.endpoints, default_model=self.args.default_model)
        task_infos = []
        for i, data in enumerate(source_datas):
            question = data['input']
            trajectory = data['output']
            answer = data['answer']
            prompt = Prompt_redundancy.format(question=question, answer=answer, trajectory=trajectory)
            for j in range(self.args.round):
                task_infos.append({
                    'sample_idx': i,
                    'resp_idx': j,
                    'prompt': prompt,
                    'session_id': f'session_{i}_{j}',
                })
        # 初始化二维结果列表：[num_samples][round]
        num_samples = len(source_datas)
        results_2d = [[None for _ in range(self.args.round)] for _ in range(num_samples)]

        # 任务队列：直接放入info字典
        task_queue = asyncio.Queue()
        for info in task_infos:
            task_queue.put_nowait(info)

        # 创建worker并发消费任务
        workers = [
            asyncio.create_task(
                self.task_worker(task_queue, results_2d, vllmclientpool)
            )
            for _ in range(self.args.concurrent_limit)
        ]

        # 进度条任务：按“样本完成度（所有round完成）”更新
        progress_task = asyncio.create_task(self._progress_monitor(results_2d, num_samples))

        await task_queue.join()

        # 让进度条任务自然结束（此时应已完成所有样本）
        await progress_task

        # 停止 worker
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # 解析 <answer>...</answer> 并按多数表决计算 redundancy（yes>no -> 1，否则 0）
        redundancy_flags = []  # 长度为 num_samples 的 0/1 列表
        for row in results_2d:
            # row = [x.choices[0].text for x in row]
            extracted = [Evaluation._to_yes_no(Evaluation._extract_answer_text(x.lower())) for x in row]
            y_cnt = sum(1 for x in extracted if x == 'yes')
            n_cnt = sum(1 for x in extracted if x == 'no')
            redundancy_flags.append(0 if y_cnt > n_cnt else 1)
        redundancy_ratio = sum(redundancy_flags) / num_samples if num_samples > 0 else 0.0

        # 打印统计结果
        print(f"No Redundancy=1 的样本比例: {redundancy_ratio:.4f} ({sum(redundancy_flags)}/{num_samples})")
        overall_path = self.args.data_path.replace('.json', '_overall.json')
        with open(overall_path, 'r', encoding='utf-8') as f:
            overall_data = json.load(f)
        overall_data['noredundancy_ratio'] = redundancy_ratio
        with open(self.args.output_path, 'w', encoding='utf-8') as f:
            json.dump(overall_data, f, ensure_ascii=False, indent=4)

    def evaluate_tool_execute_errors(self):
        source_datas = self.load_data()
        error_times = []
        tool_call_times = 0
        errors_with_correctnesses = []
        for i in tqdm(range(len(source_datas)), desc="Evaluating tool execute errors"):
            item = source_datas[i]
            output = item['output']
            if 'ikea' not in self.args.data_path.lower() and 'smart' not in self.args.data_path.lower():
                results = extract_results(output)
            else:
                if 'ikea' in self.args.data_path.lower():
                    results = extract_information(output)
                else:
                    results = extract_smart(item['steps'])
            # print(self.args.data_path)
            # results = extract_results(output)
            errors = Evaluation.check_my_result_correctness(results)
            try:
                correctness = item['metrics']['f1'] if any(dataset in self.args.data_path.lower() for dataset in ['2wiki', 'hotpotqa', 'musique', 'bamboogle']) else item['metrics']['llm_equal']
            except:
                correctness = 0
            errors_with_correctness = correctness / max(sum([not error for error in errors]), 1)
            errors_with_correctnesses.append(errors_with_correctness)
            # print('====================== errors ======================')
            # print(results)
            # print('====================== errors ======================')
            error_times.append(sum(errors))
            tool_call_times += len(results)
        total_error_count = sum(error_times)
        error_ratio = total_error_count / tool_call_times if tool_call_times > 0 else 0.0
        errors_with_correctness_ratio = sum(errors_with_correctnesses) / len(errors_with_correctnesses) if len(errors_with_correctnesses) > 0 else 0.0
        print(f"Data path: {self.args.data_path}, Tool execute correct ratio: {error_ratio:.4f} ({total_error_count}/{tool_call_times}), Errors with correctness ratio: {errors_with_correctness_ratio:.4f}")
        overall_path = self.args.data_path.replace('.json', '_overall.json')
        with open(overall_path, 'r', encoding='utf-8') as f:
            overall_metrics = json.load(f)
        overall_metrics['tool_execute_correct_ratio'] = error_ratio
        overall_metrics['errors_with_correctness_ratio'] = errors_with_correctness_ratio
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=4)

    def evaluate_thinking_length(self):
        source_datas = self.load_data()
        thinking_lengths = []
        correctness_thinking = []
        for i in tqdm(range(len(source_datas)), desc="Evaluating thinking lengths"):
            item = source_datas[i]
            output = item['output']
            if 'ikea' not in self.args.data_path.lower() and 'smart' not in self.args.data_path.lower():
                thinking = extract_thinking_results(output)
            else:
                if 'ikea' in self.args.data_path.lower():
                    thinking = extract_thinking_information(output)
                else:
                    thinking = extract_thinking_smart(item['steps'])
            thinking_length = len(self.tokenizer.encode(thinking))
            thinking_lengths.append(thinking_length)
            try:
                correctness = item['metrics']['f1'] if any(dataset in self.args.data_path.lower() for dataset in ['2wiki', 'hotpotqa', 'musique', 'bamboogle']) else item['metrics']['llm_equal']
            except:
                correctness = 0
            correctness_thinking.append(correctness / max(thinking_length, 1))
        print(f"Average thinking length: {sum(thinking_lengths) / len(thinking_lengths)}")
        print(f"Average correctness thinking: {sum(correctness_thinking) / len(correctness_thinking)}")
        overall_path = self.args.data_path.replace('.json', '_overall.json')
        with open(overall_path, 'r', encoding='utf-8') as f:
            overall_metrics = json.load(f)
        overall_metrics['thinking_length_average'] = sum(thinking_lengths) / len(thinking_lengths)
        overall_metrics['correctness_thinking_average'] = sum(correctness_thinking) / len(correctness_thinking)
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=4)

    def run(self):
        if self.tokenizer is None and self.args.model_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        if self.args.exp_type == "evaluate_redundancy":
            asyncio.run(self.evaluate_redundancy())
        elif self.args.exp_type == "evaluate_tool_execute_errors":
            self.evaluate_tool_execute_errors()
        elif self.args.exp_type == "evaluate_thinking_length":
            self.evaluate_thinking_length()
        else:
            raise ValueError(f"Invalid experiment type: {self.args.exp_type}")
            
if __name__ == "__main__":
    args = get_args()
    evaluations = Evaluation(args)
    evaluations.run()
