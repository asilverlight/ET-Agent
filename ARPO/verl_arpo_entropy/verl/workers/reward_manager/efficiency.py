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

import json
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class EfficiencyRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", sigma_tool=0.2, sigma_length=0.5) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.sigma_tool = sigma_tool
        self.sigma_length = sigma_length

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # print('================================ data non tensor batch extra info: ================================')
        # import numpy as np
        # for key, val in data.non_tensor_batch.items():
        #     if isinstance(val, list):
        #         print(f"key={key}, val={val}, type={type(val)}, shape={len(val)}")
        #     elif isinstance(val, np.ndarray):
        #         print(f"key={key}, val={val}, type={type(val)}, shape={val.shape}")
        #     else:
        #         print(f"key={key}, val={val}, type={type(val)}, shape=N/A")
        # print('================================ data non tensor batch extra info: ================================')
        # assert False

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 预处理：提前decode并把所需字段保存到新的list，便于后续处理
        preprocessed_data = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attn_mask = data_item.batch["attention_mask"]

            valid_prompt_length = attn_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"] # 这里的responses只有生成部分，没有输入prompt
            valid_response_length = attn_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            preprocessed_data.append({
                "index": i,
                "data_item": data_item,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "valid_response_length": int(valid_response_length),
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
            })

        already_print_data_sources = {}

        # 使用预处理后的数据进行打分与打印
        for entry in preprocessed_data:  # 这里的长度是一步的数据量*rollout
            i = entry["index"]
            response_str = entry["response_str"]
            prompt_str = entry["prompt_str"]
            valid_response_length = entry["valid_response_length"]
            ground_truth = entry["ground_truth"]
            data_source = entry["data_source"]
            extra_info = entry["extra_info"]

            # print(f"================================ extra_info: {extra_info} ================================")
            # assert False

            # 如果处于验证阶段（num_examine == 1），强制使用默认打分函数
            if self.num_examine == 1:
                if extra_info is None or extra_info.get("tokenizer") is None:
                    extra_info = {"tokenizer": self.tokenizer}
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            else:
                # 训练阶段：保持原有逻辑
                # add tokenizer to extra_info if not exists
                # if extra_info is None or extra_info.get("tokenizer") is None:
                #     extra_info = {"tokenizer": self.tokenizer}
                #     score = default_compute_score(
                #         data_source=data_source,
                #         solution_str=response_str,
                #         ground_truth=ground_truth,
                #         extra_info=extra_info,
                #     )
                # else:
                # 根据extra_info中的index找出其他response
                index = extra_info["index"]
                # INSERT_YOUR_CODE
                # 找出与当前entry的index相同但不是自己的其他response_str
                other_solutions = []
                other_response_lengths = []
                for other_entry in preprocessed_data:
                    if other_entry["extra_info"]["index"] == index and other_entry['response_str'] is not response_str:
                        other_solutions.append(other_entry["response_str"])
                        other_response_lengths.append(other_entry["valid_response_length"])

                # print(f'================================ other solutions =================================')
                # print(f"other_solutions: {json.dumps(other_solutions, indent=4)}")
                # print(f"other_response_lengths: {json.dumps(other_response_lengths, indent=4)}")
                # print(f'================================ other solutions =================================')
                # assert False

                # 从数据的 meta_info 读取动态缩放因子，用于按训练进度衰减 sigma
                try:
                    sigma_factor = float(data.meta_info.get("efficiency_sigma_factor", 1.0))
                except Exception:
                    sigma_factor = 1.0
                effective_sigma_tool = self.sigma_tool * sigma_factor
                effective_sigma_length = self.sigma_length * sigma_factor

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    current_response_length=valid_response_length,
                    other_solutions=other_solutions,
                    other_response_lengths=other_response_lengths,
                    sigma_tool=effective_sigma_tool,
                    sigma_length=effective_sigma_length,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print("[prompt]", prompt_str)
                # print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # print('================================ reward_tensor: ================================')
        # print(reward_tensor)
        # print('================================ reward_tensor: ================================')
        # print('================================ reward_extra_info: ================================')
        # print(reward_extra_info)
        # print('================================ reward_extra_info: ================================')
        # assert False

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "preprocessed_data": preprocessed_data,
            }
        else:
            return reward_tensor
