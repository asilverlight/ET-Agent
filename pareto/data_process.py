import json
import os
import pandas as pd
from args import get_args
import random
from tqdm import tqdm
from prompts import *
from utils import *
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

class DataProcess:
    def __init__(self, args):
        self.args = args

    def load_data(self, data_path=None):
        if data_path is None:
            data_path = self.args.data_path
        else:
            data_path = data_path
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            data = df.to_dict(orient='records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        return data

    def process_data(self):
        # 进行帕累托采样
        source_datas = self.load_data()
        # A. 数据预处理
        population = []
        for i, item in enumerate(source_datas):
            tool_use_score, correctness_score = self.calculate_metrics(item)
            item['score_tool'] = tool_use_score
            item['score_corr'] = correctness_score
            item['index'] = i
            item['domination_count'] = 0
            item['dominated_set'] = []
            population.append(item)

        # B. 快速非支配排序 (Fast Non-dominated Sort)
        fronts = [[]] # 存放各层级的前沿，fronts[0]是第一层
        
        for i in tqdm(range(len(population)), desc='快速非支配排序'):
            p = population[i]
            p['domination_count'] = 0
            p['dominated_set'] = []
            
            for q in population:
                if p['index'] == q['index']: continue
                
                # 判断 p 是否支配 q (越大越好)
                # p 支配 q 的条件：p >= q 且 p != q (至少一个严格大于)
                if (p['score_tool'] >= q['score_tool'] and p['score_corr'] >= q['score_corr']) and \
                (p['score_tool'] > q['score_tool'] or p['score_corr'] > q['score_corr']):
                    # 仅保存索引，避免对象深引用导致序列化/内存问题
                    p['dominated_set'].append(q['index'])
                # 判断 q 是否支配 p
                elif (q['score_tool'] >= p['score_tool'] and q['score_corr'] >= p['score_corr']) and \
                    (q['score_tool'] > p['score_tool'] or q['score_corr'] > p['score_corr']):
                    p['domination_count'] += 1
            
            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)

        # 依次剥洋葱，生成后续层级
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q_idx in p['dominated_set']:
                    q = population[q_idx]
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break # 没有下一层了

        # C. 收集结果 (筛选 Top N)
        final_selection = []
        items_needed = self.args.counts
        
        for front_idx, front in enumerate(fronts):
            if items_needed <= 0:
                break
                
            # 如果当前层的数据量 <= 需要的数量，全都要
            if len(front) <= items_needed:
                final_selection.extend(front)
                items_needed -= len(front)
                # print(f"第 {front_idx+1} 层 (Rank {front_idx}): 全选 {len(front)} 个")
            else:
                # 如果当前层的数据量 > 需要的数量，需要截断
                # 使用拥挤度距离排序 (距离越大越好)
                # print(f"第 {front_idx+1} 层 (Rank {front_idx}): 有 {len(front)} 个，只需 {items_needed} 个 -> 启动拥挤度筛选")
                
                sorted_front = self.calculate_crowding_distance(front)
                # 按距离降序排列 (大的在前面)
                sorted_front.sort(key=lambda x: x['distance'], reverse=True)
                
                selected_part = sorted_front[:items_needed]
                final_selection.extend(selected_part)
                items_needed = 0

        # 输出前清理临时字段，避免循环引用/无用信息
        cleaned = []
        drop_keys = {"dominated_set", "domination_count", "index"}
        for it in final_selection:
            clean_it = {k: v for k, v in it.items() if k not in drop_keys}
            # 过滤逻辑：
            # - math：correctness 同时包含 0 和 1 才保留
            # - qa  ：correctness 同时包含 >=0.75 和 <0.75 的元素才保留
            correctness = clean_it.get("correctness", None)
            src = clean_it.get("source", None)
            if correctness is None or not isinstance(correctness, list) or src is None:
                continue
            is_math = src in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo', 'still']
            if is_math:
                has_zero = any(v == 0 for v in correctness)
                has_one = any(v == 1 for v in correctness)
                if not (has_zero and has_one):
                    continue
            else:
                has_ge = any((isinstance(v, (int, float)) and v >= 0.75) for v in correctness)
                has_lt = any((isinstance(v, (int, float)) and v < 0.75) for v in correctness)
                if not (has_ge and has_lt):
                    continue
            cleaned.append(clean_it)

        with open(self.args.output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=4)

    def make_parquet_datas(self):
        # 合并出最终的rl数据
        source_datas = self.load_data()
        final_datas = []
        for i in tqdm(range(len(source_datas)), desc='合并出最终的rl数据'):
            data = source_datas[i]
            data_source = 'math_effi_rl' if data['source'] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo'] else 'qa_effi_rl'
            prompt = [
                {
                    'content': TOOLSTAR_PROMPT,
                    'role': 'system'
                },
                {
                    'content': data['input'],
                    'role': 'user'
                }
            ]
            gt = data['answer']
            if isinstance(gt, np.ndarray):
                gt = gt.tolist()
            if isinstance(gt, (str, int, float, bool)) or gt is None:
                gt = ["" if gt is None else str(gt)]
            elif isinstance(gt, (list, tuple, set)):
                gt = ["" if v is None else str(v) for v in list(gt)]
            else:
                gt = [str(gt)]

            reward_model = {'ground_truth': gt, 'style': 'rule'}
            extra_info = {
                'index': i,
                'need_tools_kwargs': True,
                'question': data['input'],
                'split': 'train'
            }
            final_datas.append({
                'data_source': data_source,
                'prompt': prompt,
                'ability': 'math' if data['source'] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo'] else 'qa',
                'reward_model': reward_model,
                'extra_info': extra_info,
                'metadata': None
            })
        random.shuffle(final_datas)
        df = pd.DataFrame(final_datas)
        df.to_parquet(self.args.output_path)

    def calculate_crowding_distance(self, layer_items):
        """
        对同一层级的点计算拥挤度距离。
        距离越大，代表该点越孤立（多样性越好），越应该被保留。
        """
        length = len(layer_items)
        if length == 0:
            return []
        
        # 初始化距离为0
        for item in layer_items:
            item['distance'] = 0.0
            
        # 针对两个目标分别计算
        for key in ['score_tool', 'score_corr']:
            # 根据当前目标排序
            layer_items.sort(key=lambda x: x[key])
            
            # 获取最大值和最小值范围
            m_min = layer_items[0][key]
            m_max = layer_items[-1][key]
            scale = m_max - m_min

            if scale == 0:
                # 此目标对拥挤度无贡献，跳过边界加权
                continue

            # 边界点的距离设为无穷大（永远优先保留边界点）
            layer_items[0]['distance'] = float('inf')
            layer_items[-1]['distance'] = float('inf')
            
            # 计算中间点的距离
            for i in range(1, length - 1):
                dist = (layer_items[i+1][key] - layer_items[i-1][key]) / scale
                layer_items[i]['distance'] += dist
                
        return layer_items

    def calculate_metrics(self, item):
        """
        计算单个元素的两个指标：
        1. tool_use_score: tool_counts 中 total_times 的方差
        2. correctness_score: correctness 的方差
        """
        # 1. 提取 total_times 列表
        # 结构：item -> 'tool_counts' -> list of dict -> 'total_times'
        times = [t['total_times'] for t in item.get('tool_counts', [])]
        
        # 计算方差 (如果你希望样本方差，可设置 ddof=1，这里默认总体方差 ddof=0)
        # 处理空列表的情况，防止报错
        if len(times) > 0:
            tool_use_score = np.var(times)
        else:
            tool_use_score = 0.0
            
        # 2. 提取 correctness 列表并计算方差
        correctness_list = item.get('correctness', [])
        if len(correctness_list) > 0:
            correctness_score = np.var(correctness_list)
        else:
            correctness_score = 0.0
            
        return tool_use_score, correctness_score


    def run(self):
        if self.args.exp_type == 'process_pareto_data':
            self.process_data()
        elif self.args.exp_type == 'make_parquet_datas':
            self.make_parquet_datas()
        else:
            raise ValueError(f"Invalid experiment type: {self.args.exp_type}")

if __name__ == '__main__':
    args = get_args()
    data_process = DataProcess(args)
    data_process.run()