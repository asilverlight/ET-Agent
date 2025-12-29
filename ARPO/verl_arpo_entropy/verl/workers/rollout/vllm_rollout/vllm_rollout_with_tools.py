# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================
# 中文说明：本文件为vLLM Rollout with Tools实现，扩展了基础LLM推理能力，
# 使其可以在推理过程中自动检测并调用外部工具（如代码解释器、搜索引擎等），
# 并将工具结果反馈回模型继续生成。
# 文件主要包括模型及工具的加载、推理主逻辑、工具调用的并发管理和熵自适应分支策略等模块。
# =========================================================

import concurrent.futures
import importlib
import json
import logging
import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Counter

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
import math
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """根据配置动态加载工具类实例。

    参数:
        tool_config: 工具的OmegaConf配置，包含class_path和初始化参数params

    返回值:
        初始化好的工具对象（BaseTool的子类实例）
    """
    module_path, class_name = tool_config.class_path.rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)  # 动态导入模块
        tool_class = getattr(module, class_name)
        if 'search' in class_name.lower():
            if tool_config.use_local_search:
                tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True).get('localsearch', {})
            else:
                tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True).get('bingsearch', {})
        else:
            tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        tool_instance = tool_class(**tool_params)
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    高级版vLLM生成引擎。在文本生成过程中，模型能通过特定token触发外部工具（如代码执行器、搜索引擎等），
    工具结果会反馈回继续生成，支持自适应beaming策略和熵控制，用于丰富模型能力与生成多样性。
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """
        初始化vLLMRolloutWithTools
        参数说明：
            model_path      —— 基座模型目录
            config          —— OmegaConf参数，包含生成相关参数和工具相关配置
            tokenizer       —— 分词器对象
            model_hf_config —— huggingface模型配置
        """
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        # 从配置中获取beam search和聚类采样相关参数
        self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])   # 初始每个样本rollout数
        self.beam_size = self.config.get("beam_size", 1)                                # 最大分支数
        self.branch_probability = self.config.get("branch_probability", 0.5)            # 分支概率门限
        self.entropy_weight = self.config.get("entropy_weight", 0.5)                    # 熵变化分支惩罚系数
        
        # 解析工具相关配置（dict，嵌套"tool_instances"）
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # 通用工具调用上限和资源限制
        self.tool_call_limit = tools_config.get("call_limit", 5)                # 每个样本最大可调用某工具次数
        self.max_tool_workers = tools_config.get("max_workers", 32)             # 最大并发工具工作线程数
        self.tool_timeout = tools_config.get("timeout", 120)                    # 工具超时时间（秒）

        # 工具重试/日志配置等
        self.tool_retry_count = tools_config.get("retry_count", 3)              # 工具最大重试次数
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)  # 工具详细日志

        # ========== 加载所有配置的工具 ==========
        self.tools: Dict[str, BaseTool] = {}     # {trigger_tag: tool_instance}
        
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)        # 动态加载工具
                    self.tools[tool_instance.trigger_tag] = tool_instance      # 按触发tag记录
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        # tool调用后，遇到</tag>作为stop sequence，终止生成并转工具
        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 10 # 用于熵计算的logprob数目
        self.initial_entropy_dict = {}  # 记录每个active indice的初始熵（生成分支用）

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured.")

        # 工具调用线程池（最大max_tool_workers并发）
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

    def __del__(self):
        # 析构时关闭线程池，终止后台工具任务
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str) -> str:
        """
        从text中抽取<tag>和</tag>之间的内容（通常用于获取工具输入）。
        只取text中最后一个<tag>...内容...</tag>片段。
        若无，则返回""
        """
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content):
        """
        工具调用带重试及超时机制。多次失败也不会抛异常，最多尝试self.tool_retry_count次
        返回dict，包括success/retry_count/execution_time/result四项
        """
        retry_count = 0
        start_time = time.time()
        success = False
        
        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text:
                    success = True
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text
                    }
                else:
                    logger.warning(f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}")
                    retry_count += 1
            except Exception as e:
                logger.error(f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                retry_count += 1
        
        execution_time = time.time() - start_time
        logger.warning(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": ""
        }

    def _calc_entropy(self, logprobs):
        """
        用于根据一段logprobs计算分支的熵值
        熵的高低反映当前分布的不确定度，可用于智能控制生成分支多样性（分支概率）
        """
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        vLLMRolloutWithTools主流程（入口函数）：
        1. 多轮推理，每轮检测是否触发工具、管理分支、控制工具并发和采集指标。
        2. 工具能响应模型生成的<tag>...</tag>内容，回写到当前分支输入序列。
        3. 支持自适应分支生长以及生成长度、掩码、统计metrics等。
        """
        # ========== 步骤1：预处理输入/参数 ==========
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)
        
        # 工具调用全局统计（用于收集性能与故障指标，便于评估工具调用链瓶颈等）
        tool_metrics = {
            "tools/total_calls": 0,             # 全部工具调用次数
            "tools/successful_calls": 0,        # 成功调用数
            "tools/failed_calls": 0,            # 失败（无结果或异常）调用数
            "tools/total_execution_time": 0.0,  # 汇总执行时间
            "tools/avg_execution_time": 0.0,    # 平均每次工具调用耗时
            "tools/max_execution_time": 0.0,    # 单次最大工具耗时
            "tools/max_retries": 0,             # 单次最大重试次数
            "tools/total_retries": 0,           # 累计重试总数
            "tools/call_limit_reached_count": 0,# 样本达到调用上线次数
        }
        
        # 每个工具（tag维度）的独立统计Counter
        calls_per_tool = Counter()      # 工具A/B/C调用次数
        success_per_tool = Counter()    # 工具A/B/C成功次数
        total_time_per_tool = Counter() # 工具A/B/C累计耗时

        do_sample = prompts.meta_info.get('do_sample', True)         # 是否采样生成（非贪婪解码）
        is_validate = prompts.meta_info.get('validate', False)       # 校验模式/评估模式
        # ========== 步骤2：采样参数与Beam相关配置 ==========
        beam_size = self.beam_size
        if not do_sample:
            # 若为贪婪生成，固定参数，关闭采样相关参数，分支数=1
            kwargs.update({
                'best_of': 1, 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'temperature': 0, 'n': 1
            })
            beam_size = 1
        elif is_validate:
            # 校验时用更收敛的采样参数及分支数1
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1  # 验证模式只要一个样本
            })
            beam_size = 1
        
        # 临时修复OOV采样异常：强制只允许合法词表
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n

            # 对每个输入实例预处理，得到prompt的token id列表（可pad补齐）
            prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids]

            # === 对每个样本展开若干个初始rollout，为后续多分支/beam做准备 ===
            initial_rollouts = self.initial_rollouts
            initial_rollouts = min(initial_rollouts, num_samples)  # 初始分支不能超过样本请求分支总数

            curr_inputs = []        # 对应每个活跃分支的输入token序列
            init_inputs = []        # 记录每个分支最初prompt token
            result_masks = []       # 掩码，标记哪些token来自工具调用插入
            call_counters = []      # 当前sample已调用工具次数
            active_indices = []     # 当前所有活跃分支索引
            
            # 构造初始每个样本的多份rollout
            for i, ids in enumerate(prompt_token_ids_list):
                for _ in range(initial_rollouts):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)
            
            # 统计每个原始样本拥有多少分支/副本
            rollouts_per_sample = [initial_rollouts] * batch_size
            sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}

            max_len = self.config.response_length  # 最大生成响应长度

            # =========== 进入生成循环：只要当前仍有活跃分支就继续生成 ===========
            while active_indices:
                active_prompts = [curr_inputs[i] for i in active_indices]
                # logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                # logger.debug(f"active_indices: {active_indices}")
                # logger.debug(f"active_prompts: {active_prompts}")

                # 更新本轮采样分支最大可生成token数
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs = self.logprobs
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                # ========== 熵变化自适应分支机制监控 ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                current_entropy_dict = {}
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    logprobs = []
                    tokens = output.outputs[0].token_ids
                    for j in range(min(20, len(tokens))):
                        try:
                            logprob_info = output.outputs[0].logprobs[j]
                        except Exception:
                            logprob_info = output.outputs[0].logprobs[-1]
                        token_list = list(logprob_info.values())
                        token_logprobs = [token.logprob for token in token_list]
                        logprobs.extend(token_logprobs)
                    if logprobs:
                        entropy = self._calc_entropy(logprobs) / entropy_norm_factor
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    if out_idx not in self.initial_entropy_dict:
                        self.initial_entropy_dict[out_idx] = entropy
                # ===========================

                # 工具请求队列，结构：{tag: List[{"index":分支序号, "content":内容}]}
                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids
                    

                    curr_inputs[out_idx].extend(generated_tokens)                  # 拼接生成token到原输入
                    result_masks[out_idx].extend([1] * len(generated_tokens))      # 更新mask

                    finish_reason = output.outputs[0].finish_reason                # 终止原因
                    stop_reason = output.outputs[0].stop_reason                    # 触发终止的stop token

                    is_tool_call = finish_reason == 'stop' and stop_reason in self.stop_sequences
                    
                    # Debug日志：输出当前生成内容及终止信息
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    # logger.debug(f"  Sample {out_idx} output:")
                    # logger.debug(f"  Token IDs: {generated_tokens}")
                    # logger.debug(f"  Text: {decoded_text}")
                    # logger.debug(f"  Finish reason: {finish_reason}")
                    # logger.debug(f"  Stop reason: {stop_reason}")
                    # logger.debug(f"  Is tool call: {is_tool_call}")
                    # logger.debug(f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}")

                    # ==== 处理是否触发外部工具 ====
                    if is_tool_call:
                        tag = stop_reason.strip("</>")  # 提取工具tag名
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)  # 取<tag>内容
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 统计工具调用
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            curr_inputs[out_idx].append(eos_token_id)         # 达到调用上线后强制终止
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == 'length':
                        # 若生成到最大长度但未结束，继续本分支
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)

                    elif finish_reason == 'stop':  # EOS
                        pass

                # ==== 用并发池执行所有工具请求，结果写回分支 ====
                if any(tool_requests.values()):
                    # print(f'=========tool_requests=========')
                    # for key, value in tool_requests.items():
                    #     print(f"Tool: {key}, Requests: {json.dumps(value, indent=4)}")
                    # print(f'=========tool_requests=========')
                    # assert False
                    logger.info(f"Processing tool requests: {sum(len(reqs) for reqs in tool_requests.values())} total requests")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        # logger.debug(f"Processing {len(requests)} requests for tool '{tag}'")
                        tool = self.tools[tag]
                        for req in requests:
                            # logger.debug(f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}")
                            future = self.executor.submit(self._execute_tool_with_retry, tool, req["content"])
                            futures[future] = {"index": req["index"], "tag": tag}

                    total_futures = len(futures)
                    completed_futures = 0
                    # logger.debug(f"Submitted {total_futures} tool requests for execution")
                    for future in concurrent.futures.as_completed(futures):
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # 工具执行结果解包
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]
                            
                            # 统计
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                # logger.info(f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}")
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = f"Tool({tag}) returned empty output."
                                # logger.warning(f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s")
                            
                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            total_time_per_tool[tag] += execution_time  # 工具tag维统计
                            
                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                # logger.warning(f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s")
                            else:
                                # logger.debug(f"Tool({tag}) result: {result_text}")
                                pass
                                
                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                        
                        # logger.debug(f"Tool completion progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)")
                        # 工具结果格式化打包为<result>块写回输入
                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result)
                        # logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))
                # assert False
                # ========== 进入下一轮 ==== 按生成长度与分支规则继续维护活跃分支 ==========
                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)
                
                # ==== Beam Search/分支机制入口 ====
                new_indices = []
                new_inputs = []
                new_init_inputs = []
                new_result_masks = []
                new_call_counters = []
                new_sample_origins = []  # 记录每个新分支对应的原始样本
                
                # Map from original sample index to its active rollouts
                active_by_sample = {}
                for idx in final_active_indices:
                    # 查询本分支属于哪个原始样本
                    orig_sample = None
                    for sample_idx, indices in sample_to_indices.items():
                        if idx in indices:
                            orig_sample = sample_idx
                            break
                    
                    if orig_sample is not None:
                        if orig_sample not in active_by_sample:
                            active_by_sample[orig_sample] = []
                        active_by_sample[orig_sample].append(idx)
                
                # 已活跃的每个原始样本做beam，尽量复刻到指定分支数
                for orig_sample, active_idxs in active_by_sample.items():
                    remaining_slots = num_samples - rollouts_per_sample[orig_sample]
                    if remaining_slots <= 0:
                        continue
                    branches_created = 0
                    for source_idx in active_idxs:
                        branches_per_idx = min(beam_size - 1, remaining_slots - branches_created)
                        if branches_per_idx <= 0:
                            break
                        for _ in range(branches_per_idx):
                            # ==== 熵变化自适应分支控制 ====
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                            entropy_delta = entropy_now - entropy_init
                            prob = random.random() - self.entropy_weight * entropy_delta  # 熵越高分支更难
                            prob = max(0.0, min(1.0, prob))   # 限制在[0,1]
                            if prob > self.branch_probability: 
                                continue
                            # ==== END ====
                            new_inputs.append(curr_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append(result_masks[source_idx].copy())
                            new_call_counters.append(call_counters[source_idx])
                            new_sample_origins.append(orig_sample)
                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                        if branches_created >= remaining_slots:
                            break

                # 把不活跃但分支数不够的样本也补足
                for orig_sample in range(batch_size):
                    if orig_sample not in active_by_sample and rollouts_per_sample[orig_sample] < num_samples:
                        # 针对不活跃的样本，每次最多只新增一个分支
                        branches_to_add = min(1, num_samples - rollouts_per_sample[orig_sample])
                        if branches_to_add <= 0:
                            continue
                            
                        # 以第一个分支为模板（可按需调整）
                        source_idx = sample_to_indices[orig_sample][0]
                        # 新建分支
                        for _ in range(branches_to_add):
                            new_inputs.append(init_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append([])
                            new_call_counters.append(0)
                            new_sample_origins.append(orig_sample)  # 记录原始样本
                            rollouts_per_sample[orig_sample] += 1
                
                # 批量将新分支添加至当前活跃集合
                if new_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(new_inputs)
                    init_inputs.extend(new_init_inputs)
                    result_masks.extend(new_result_masks)
                    call_counters.extend(new_call_counters)
                    # 新分支加入活跃序列
                    final_active_indices.extend(range(start_idx, start_idx + len(new_inputs)))
                    # 原始样本映射更新
                    for i, new_idx in enumerate(range(start_idx, start_idx + len(new_inputs))):
                        orig_sample = new_sample_origins[i]
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                
                active_indices = final_active_indices

            # ========== 收敛后格式化输出 ==========
            # 所有分支都已终止时，裁剪超长序列
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][:offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]
            
            # 根据原始样本分组，确保每个样本有num_samples份输出（不足补足）
            output_sequences = []
            output_result_masks = []
            for i in range(batch_size):
                # 提取该样本下的所有分支索引
                sample_indices = sample_to_indices.get(i, [])
                # 按需选前num_samples个，没有则取最后一个补齐
                selected_indices = sample_indices[:num_samples]
                while len(selected_indices) < num_samples:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        break  # 不存在的极端情况
                for idx in selected_indices:
                    output_sequences.append(curr_inputs[idx][len(prompt_token_ids_list[i]):])  # 取生成内容
                    output_result_masks.append(result_masks[idx])                             # 掩码

            # 补齐pad输出到最大长度
            padded_response_list = []
            padded_result_mask_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                # logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)
            
            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            
            # non_tensor_batch（非tensor型附加字段）补齐复制
            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            if num_samples > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, num_samples)
                attention_mask = _repeat_interleave(attention_mask, num_samples)
                position_ids = _repeat_interleave(position_ids, num_samples)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(value, num_samples, axis=0)
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [item for item in value for _ in range(num_samples)]

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

            # 兼容特殊位置编码（如Qwen2的三维RoPE分段递增），为所有分支追加正确position_id
            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            loss_mask = loss_mask * response_attention_mask  # 掩码只用于有效的响应区域

            # ========== 工具指标归档与汇总 ==========
            # 计算平均耗时
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
                
            # 逐工具统计平均耗时和成功率等
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            # 打包为TensorDict作为batch输出
            batch = TensorDict({
                "prompts": input_ids,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "loss_mask": loss_mask,
                "position_ids": final_position_ids,
            }, batch_size=final_batch_size)

        # 内存自动回收机制：如果配置VLLM为0.5.4等特定版本需特殊处理
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            
        # 合并所有metrics，汇总写到meta_info中返回
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        return data_proto