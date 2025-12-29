import time
import hashlib
from .utils import *
from .prompts import *
from .src_sample_process import *
from .process_input import *  # pyright: ignore[reportMissingImports]
import re
import json
import asyncio

class SampleProcessor:
    def __init__(
        self,
        prompt_manager,
        tool_executor,
        vllm_pool_m1, # m17
        vllm_pool_m2, # 4b
        vllm_pool_m3, # 32b
        tokenizer_m1,
        tokenizer_m2,
        tokenizer_m3,
        args,
        sample_stat,
        session_id,
    ):
        self.prompt_manager = prompt_manager
        self.tool_executor = tool_executor
        self.vllm_pool_m1 = vllm_pool_m1
        self.vllm_pool_m2 = vllm_pool_m2
        self.vllm_pool_m3 = vllm_pool_m3
        self.tokenizer_m1 = tokenizer_m1
        self.tokenizer_m2 = tokenizer_m2
        self.tokenizer_m3 = tokenizer_m3
        self.args = args
        self.use_log = args.use_log
        self.compatible_search = args.compatible_search
        self.use_local_search = args.use_local_search
        self.max_search_result_length = args.max_search_result_length
        self.max_python_result_length = args.max_python_result_length
        self.sample_stat = sample_stat
        self.question = sample_stat.get("question", sample_stat["input"])
        # system_prompt = self.prompt_manager.get_system_prompt(self.format)

        if not session_id:
            session_content = f"{ToolStar_Prompt}_{self.question}"
            session_id = hashlib.md5(session_content.encode()).hexdigest()
        self.session_id = session_id

        self.sample_start_time = None
        self.llm_time = 0
        self.python_time = 0
        self.search_time = 0
        self.total_time = None
        self.python_rounds = 0
        self.search_rounds = 0
        self.initial_prompt = ''
        self.generate_initial_trajectory_context = ''
        self.judge_correctness_context = ''
        self.correct_wrong_trajectory_context = ''
        self.evolve_stragety_context = ''

    async def generate_initial_trajectory(self, trajectory=None, python_times=0, search_times=0, max_python_times=None, max_search_times=None, input=None):
        generate_initial_trajectory_context = input
        if trajectory is None:
            outputs, logs = '', []
        else:
            outputs = trajectory
            logs = split_trajectory_to_logs(trajectory)
        if max_python_times is None:
            max_python_times = self.args.max_python_times
        if max_search_times is None:
            max_search_times = self.args.max_search_times
        if input is None: # 默认进行最原始的推理
            generate_initial_trajectory_context = process_generate_initial_trajectory_input(self.tokenizer_m1, self.sample_stat["input"])
        while True:
            if self.use_log:
                print('Current Prompt: ================================')
                print(generate_initial_trajectory_context)
                print('===============================================')
            output, outputs, logs, generate_initial_trajectory_context = await call_llm_for_generate_trajectory(self.vllm_pool_m1, generate_initial_trajectory_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, outputs, logs)
            if not output:
                break
            tool_tag = self.tool_executor.identify_tool(output)
            if tool_tag == "python" and python_times < max_python_times:
                _, (outputs, logs, generate_initial_trajectory_context) = await call_python(self.tool_executor, self.tool_executor.extract_content(output, "python"), outputs, logs, generate_initial_trajectory_context, use_log=self.use_log, max_python_result_length=self.max_python_result_length)
                python_times += 1
            elif tool_tag == "search" and search_times < max_search_times:
                _, (outputs, logs, generate_initial_trajectory_context) = await call_search(self.tool_executor, self.tool_executor.extract_content(output, "search"), outputs, logs, generate_initial_trajectory_context, source=self.sample_stat["source"], use_log=self.use_log, compatible_search=self.compatible_search, use_local_search=self.use_local_search)
                search_times += 1
            else:
                if not output.strip().endswith("</answer>"):
                    output, outputs, logs, generate_initial_trajectory_context = await call_llm_for_generate_trajectory(self.vllm_pool_m1, generate_initial_trajectory_context, False, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, outputs, logs)
                    if not output:
                        break
                else:
                    break
        prediction = extract_answer(outputs)
        return outputs, logs, prediction

    async def local_evolve_correct_trajectory(self, evolve_trajectory_context, best_trajectory):
        evolve_trajectory_result = await call_local_llm_for_generate_trajectory(self.vllm_pool_m3, evolve_trajectory_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, use_log=self.use_log)
        self.sample_stat["evolve_judgements"].append(f'Correct, refine on step level: \n{evolve_trajectory_result}')

        if evolve_trajectory_result:
            lower_result = evolve_trajectory_result.lower()
            if 'step: no' in lower_result:
                return None
            corrected_step_key = "corrected step:"
            analysis_key = "analysis:"
            step_key = "step:"
            corrected_step_idx = lower_result.find(corrected_step_key)
            analysis_step_idx = lower_result.find(analysis_key)
            step_part, corrected_step_content = None, None
            if corrected_step_idx != -1:
                # 前半部分：从最后一个"Step:"到"Corrected Step:"前
                before_corrected = lower_result[:analysis_step_idx]
                last_step_idx = before_corrected.find(step_key)
                if last_step_idx != -1:
                    # step_part从原始字符串中切片，保持原始大小写
                    step_part = evolve_trajectory_result[last_step_idx:analysis_step_idx].strip()
                else:
                    step_part = ""
                # 后半部分："Corrected Step:"后面的内容
                corrected_step_content = evolve_trajectory_result[
                    corrected_step_idx + len(corrected_step_key) :
                ].strip()
            if step_part and corrected_step_content:
                # 从step_part中提取步数，并转为int型
                step_num = None
                match = re.search(r"step:\s*(\d+)", step_part, re.IGNORECASE)
                if match:
                    step_num = int(match.group(1))
                else:
                    return None
                trajectory = best_trajectory
                # 用正则分割出每个步骤（以</result>为分界，包含</result>，但不包含后续的内容）
                step_pattern = re.compile(r'(.*?</result>)', re.DOTALL)
                steps = step_pattern.findall(trajectory)
                # 如果最后还有残留内容（如<answer>），也加进去
                rest = step_pattern.sub('', trajectory)
                if rest.strip():
                    steps.append(rest)
                # step_num是第几个步骤（从1开始），找到并替换
                if 1 <= step_num <= len(steps):
                    steps[step_num - 1] = corrected_step_content
                # 合并回去
                # 只保留第step_num个步骤及其之前的内容，后面的步骤全部去除
                new_trajectory = ''.join(steps[:step_num])
                tool_execute_result = await self.execute_tool_result(corrected_step_content, new_trajectory)
                new_trajectory += tool_execute_result
                if not new_trajectory.strip().endswith("</answer>"):
                    python_times, search_times = calculate_python_times(new_trajectory), calculate_search_times(new_trajectory)
                    direct_generate_input = process_modify_previous_trajectory_input(self.tokenizer_m1, self.sample_stat["input"], new_trajectory)
                    new_trajectory, logs, _ = await self.generate_initial_trajectory(
                        trajectory=new_trajectory,
                        python_times=python_times,
                        search_times=search_times,
                        max_python_times=self.args.max_python_times+self.args.additional_python_times,
                        max_search_times=self.args.max_search_times+self.args.additional_search_times,
                        input=direct_generate_input,
                    )

                return new_trajectory
        return None

    async def judge_correctness(self, question=None, labeled_answer=None, pred_answer=None):
        judge_result = False
        if self.sample_stat["source"] not in ['openr1', 'numina-tir', 'numina-cot', 'aime']: # 是qa问题
            if not isinstance(labeled_answer, list):
                labeled_answers = [labeled_answer]
            else:
                labeled_answers = labeled_answer
            judge_result = True if max([calculate_f1_score(pred_answer, labeled_answer) for labeled_answer in labeled_answers]) > 0.8 else False
        else:
            judge_correctness_context = process_judge_correctness_input(self.tokenizer_m3, question, labeled_answer, pred_answer)
            judge_result = await call_local_llm_for_judge_correctness(self.vllm_pool_m3, self.args.sampling_params, judge_correctness_context, self.session_id)
        return judge_result

    async def execute_tool_result(self, output, current_trajectory):
        logs = split_trajectory_to_logs(current_trajectory)
        tool_tag = self.tool_executor.identify_tool(output)
        tool_execute_result = ''
        if tool_tag == "python":
            tool_execute_result, _ = await call_python(self.tool_executor, self.tool_executor.extract_content(output, "python"), use_log=self.use_log, max_python_result_length=self.max_python_result_length)

        elif tool_tag == "search":
            _, (tool_execute_result, _, _) = await call_search(self.tool_executor, self.tool_executor.extract_content(output, "search"), '', logs, '', source=self.sample_stat["source"], use_log=self.use_log, compatible_search=self.compatible_search, use_local_search=self.use_local_search)
        return tool_execute_result

    async def refine_trajectory_whole_level(self, trajectory):
        # 在正确的情况下，对整条链做重写
        initial_input = process_input_correct_rewrite_whole_trajectory(self.tokenizer_m3, self.sample_stat["input"], trajectory)
        actions = split_trajectory_actions(trajectory)
        if len(actions) == 0:
            output = await call_local_llm_for_naive_generation(self.vllm_pool_m3, self.args.sampling_params, initial_input, self.session_id)
            if not output.startswith("<think>"):
                output = "<think>" + output
            return output
        refined_trajectory = '<think>'
        for i in range(len(actions)):
            output = await call_local_llm_for_naive_generation(self.vllm_pool_m3, self.args.sampling_params, initial_input, self.session_id)
            if '</think>' in output:
                output = output.split('</think>')[0] + '</think>'
            initial_input += output
            initial_input += actions[i]
            refined_trajectory += output + actions[i]
        self.sample_stat['evolve_judgements'].append('Correct, rewrite whole trajectory')
        return refined_trajectory

    async def find_refine_step(self, trajectory):
        # 当答错时, 找出需要修正的步骤
        evolve_incorrect_trajectory_refine_step_context = process_input_incorrect_step_refine(self.tokenizer_m3, self.sample_stat["input"], trajectory)
        if 'Analysis:' not in self.args.sampling_params.stop:
            self.args.sampling_params.stop.append('Analysis:')
        evolve_incorrect_trajectory_refine_step_result = await call_local_llm_for_generate_trajectory(self.vllm_pool_m3, evolve_incorrect_trajectory_refine_step_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, use_log=self.use_log)
        if 'Analysis:' in self.args.sampling_params.stop:
            self.args.sampling_params.stop.remove('Analysis:')
        if evolve_incorrect_trajectory_refine_step_result is None:
            return None
        
        # 提取是第几步骤
        step_key = 'step:'
        analysis_key = 'analysis:'
        lower_evolve_incorrect_trajectory_refine_step_result = evolve_incorrect_trajectory_refine_step_result.lower()
        step_idx = lower_evolve_incorrect_trajectory_refine_step_result.find(step_key)
        if 'analysis:' not in lower_evolve_incorrect_trajectory_refine_step_result:
            lower_evolve_incorrect_trajectory_refine_step_result += 'analysis:'
        analysis_idx = lower_evolve_incorrect_trajectory_refine_step_result.find(analysis_key)
        if step_idx == -1 or analysis_idx == -1:
            return None
        step_content = lower_evolve_incorrect_trajectory_refine_step_result[step_idx:analysis_idx].strip()
        step_num = None
        match = re.search(r"step:\s*(\d+)", step_content, re.IGNORECASE)
        if match:
            step_num = int(match.group(1))
        else:
            return None
        return step_num

    async def evolve_incorrect_trajectory_refine_step(self, trajectory, step_num):
        # 对单步做refine
        step_pattern = re.compile(r'(.*?</result>)', re.DOTALL)
        steps = step_pattern.findall(trajectory)
        rest = step_pattern.sub('', trajectory)
        if rest.strip():
            steps.append(rest)
        steps = [step.strip() for step in steps]
        if 1 <= step_num <= len(steps):# 正确找到了需要修改的步骤
            # 把前step_num个step拼起来
            new_trajectory = ''.join(steps[:step_num])
            evolve_incorrect_trajectory_refine_step_context = process_input_incorrect_step_refine(self.tokenizer_m2, self.sample_stat["input"], new_trajectory)
            evolve_incorrect_trajectory_refine_step_context += f'\n\nStep: {step_num}\nAnalysis: Let me analyze this step.'
            analysis_content = await call_local_llm_for_generate_trajectory(self.vllm_pool_m2, evolve_incorrect_trajectory_refine_step_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, use_log=self.use_log)
            # analysis_content是修正的建议

            # 接下来生成新的步骤
            new_step_context = process_input_incorrect_generate_new_step(self.tokenizer_m2, self.sample_stat["input"], new_trajectory, analysis_content)
            new_step_result = await call_local_llm_for_generate_trajectory(self.vllm_pool_m2, new_step_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, use_log=self.use_log)
            self.sample_stat['evolve_judgements'].append(f'Incorrect, generate new step: \nRefine step {step_num}: \n{analysis_content}\nNew step: \n{new_step_result}')
            new_step = new_step_result.strip()
            if not new_step.startswith('<think>'):
                new_step = '<think>' + new_step
            # 如果new_step中有<search>或<python>或<answer>，则在其前面插入</think>
            special_tags = ['<search>', '<python>', '<answer>']
            # 查找第一个出现的特殊标识符及其位置
            first_idx = len(new_step)
            tag_to_insert = None
            for tag in special_tags:
                idx = new_step.find(tag)
                if idx != -1 and idx < first_idx:
                    first_idx = idx
                    tag_to_insert = tag
            if tag_to_insert is not None and not new_step[:first_idx].rstrip().endswith('</think>'):
                # 在first_idx前插入</think>
                new_step = new_step[:first_idx] + '</think>' + new_step[first_idx:]
            steps[step_num - 1] = new_step
            new_trajectory = ''.join(steps[:step_num])

            # 生成新的工具调用
            tool_execute_result = await self.execute_tool_result(new_step, new_trajectory)
            new_trajectory += tool_execute_result

            if not new_trajectory.strip().endswith("</answer>"):
                python_times, search_times = calculate_python_times(new_trajectory), calculate_search_times(new_trajectory)
                direct_generate_input = process_modify_previous_trajectory_input(self.tokenizer_m1, self.sample_stat["input"], new_trajectory)
                new_trajectory, logs, _ = await self.generate_initial_trajectory(
                    trajectory=new_trajectory,
                    python_times=python_times,
                    search_times=search_times,
                    max_python_times=self.args.max_python_times+self.args.additional_python_times,
                    max_search_times=self.args.max_search_times+self.args.additional_search_times,
                    input=direct_generate_input,
                )
            return new_trajectory
        self.sample_stat['evolve_judgements'].append(f'Step number {step_num} is incorrect')
        return None

    async def evolve_incorrect_trajectory_hint_step(self, trajectory, step_num):
        # 对单步做hint
        step_pattern = re.compile(r'(.*?</result>)', re.DOTALL)
        steps = step_pattern.findall(trajectory)
        steps = [step.strip() for step in steps]
        if 1 <= step_num <= len(steps):# 正确找到了需要修改的步骤
            # 把前step_num个step拼起来
            if self.sample_stat["source"] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo']:
                analysis_content = random.choice(Python_Hints)
            else:
                analysis_content = random.choice(Search_Hints)
            self.sample_stat['evolve_judgements'].append(f'Incorrect, hint on step level: \nHint step {step_num}: \n{analysis_content}')
            new_trajectory = ''.join(steps[:step_num])
            if new_trajectory.strip().endswith('</result>'):
                # 情况一，直接拼接 <think>analysis_content
                new_trajectory = new_trajectory.rstrip() + '<think>' + random.choice(analysis_content)
            elif new_trajectory.strip().endswith('</answer>'):
                # 去掉所有 <answer> 和 </answer>
                new_trajectory_no_answer = new_trajectory.replace('<answer>', '').replace('</answer>', '')
                # 从后往前找到最后一个 </result>
                last_result_idx = new_trajectory_no_answer.rfind('</result>')
                if last_result_idx != -1:
                    after_result = new_trajectory_no_answer[last_result_idx + len('</result>'):]

                    # 查找after_result中有无<think>
                    think_idx = after_result.find('<think>')
                    if think_idx != -1:
                        # 有<think>，看<think>后面有没有</think>
                        think_close_idx = after_result.find('</think>', think_idx + len('<think>'))
                        if think_close_idx != -1:
                            # 删掉</think>
                            after_result = after_result[:think_close_idx] + after_result[think_close_idx + len('</think>'):]
                    else:
                        # 没有<think>，就在</result>后加一个<think>
                        after_result = '<think>' + after_result

                    # 拼接前半部分+处理后的after_result+analysis_content
                    new_trajectory = new_trajectory_no_answer[:last_result_idx + len('</result>')] + after_result + analysis_content
                else:
                    # 如果没找到</result>，直接拼接
                    new_trajectory = new_trajectory_no_answer + analysis_content
            python_times, search_times = calculate_python_times(new_trajectory), calculate_search_times(new_trajectory)
            direct_generate_input = process_modify_previous_trajectory_input(self.tokenizer_m1, self.sample_stat["input"], new_trajectory)
            new_trajectory, logs, _ = await self.generate_initial_trajectory(
                trajectory=new_trajectory,
                python_times=python_times,
                search_times=search_times,
                max_python_times=self.args.max_python_times+self.args.additional_python_times,
                max_search_times=self.args.max_search_times+self.args.additional_search_times,
                input=direct_generate_input,
            )
            return new_trajectory
        self.sample_stat['evolve_judgements'].append(f'Step number {step_num} is incorrect')
        return None

    async def evolve_incorrect_trajectory_step_level(self, trajectory):
        step_num = await self.find_refine_step(trajectory)
        if step_num is None:
            return [None]
        task_refine_step = asyncio.create_task(
            self.evolve_incorrect_trajectory_refine_step(trajectory, step_num)
        )
        task_hint_step = asyncio.create_task(
            self.evolve_incorrect_trajectory_hint_step(trajectory, step_num)
        )
        new_evolved_trajectory_refine_step, new_evolved_trajectory_hint_step = await asyncio.gather(
            task_refine_step, task_hint_step
        )
        return [new_evolved_trajectory_refine_step, new_evolved_trajectory_hint_step]

    async def evolve_incorrect_trajectory_refine_trajectory(self, trajectory):
        # 在错误的情况下，对于整个轨迹做refine
        evolve_incorrect_trajectory_refine_trajectory_context = process_input_incorrect_insert_analysis_tail(self.tokenizer_m2, self.sample_stat["input"], trajectory)

        evolve_incorrect_trajectory_refine_trajectory_result = await call_local_llm_for_generate_trajectory(self.vllm_pool_m2, evolve_incorrect_trajectory_refine_trajectory_context, False, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, use_log=self.use_log)
        self.sample_stat['evolve_judgements'].append(f'Incorrect, refine whole trajectory: \n{evolve_incorrect_trajectory_refine_trajectory_result}')
        new_trajectory = ''
        if trajectory.strip().endswith('</answer>'):
            new_trajectory = trajectory.replace('<answer>', '').replace('</answer>', '')
        last_result_idx = new_trajectory.rfind('</result>')
        if last_result_idx != -1:
            after_result = new_trajectory[last_result_idx + len('</result>'):]
        else:
            after_result = new_trajectory
        think_idx = after_result.find('<think>')
        if think_idx != -1:
            think_close_idx = after_result.find('</think>', think_idx + len('<think>'))
            if think_close_idx != -1:
                after_result = after_result[:think_close_idx] + after_result[think_close_idx + len('</think>'):]
        else:
            if last_result_idx == -1:
                new_trajectory = '<think>' + after_result
            else:
                after_result = '<think>' + after_result
        if last_result_idx != -1:
            new_trajectory = new_trajectory[:last_result_idx + len('</result>')] + after_result
        else:
            new_trajectory = after_result
        new_trajectory += evolve_incorrect_trajectory_refine_trajectory_result
        if evolve_incorrect_trajectory_refine_trajectory_result is None:
            return None
        python_times, search_times = calculate_python_times(new_trajectory), calculate_search_times(new_trajectory)
        direct_generate_input = process_modify_previous_trajectory_input(self.tokenizer_m1, self.sample_stat["input"], new_trajectory)
        new_trajectory, logs, _ = await self.generate_initial_trajectory(
            trajectory=new_trajectory,
            python_times=python_times,
            search_times=search_times,
            max_python_times=self.args.max_python_times+self.args.additional_python_times,
            max_search_times=self.args.max_search_times+self.args.additional_search_times,
            input=direct_generate_input,
        )
        return new_trajectory

    async def evolve_incorrect_trajectory_hint_trajectory(self, trajectory):
        # 在错误的情况下，在轨迹的最后插入hint
        if self.sample_stat["source"] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo']:
            hint_content = random.choice(Python_Hints)
        else:
            hint_content = random.choice(Search_Hints)
        self.sample_stat['evolve_judgements'].append(f'Incorrect, hint on trajectory level: \nHint on trajectory level: \n{hint_content}')
        new_trajectory = ''
        if trajectory.strip().endswith('</answer>'):
            new_trajectory = trajectory.replace('<answer>', '').replace('</answer>', '')
        last_result_idx = new_trajectory.rfind('</result>')
        if last_result_idx != -1:
            after_result = new_trajectory[last_result_idx + len('</result>'):]
        else:
            after_result = new_trajectory
        think_idx = after_result.find('<think>')
        if think_idx != -1:
            think_close_idx = after_result.find('</think>', think_idx + len('<think>'))
            if think_close_idx != -1:
                after_result = after_result[:think_close_idx] + after_result[think_close_idx + len('</think>'):]
        else:
            if last_result_idx == -1:
                new_trajectory = '<think>' + after_result
            else:
                after_result = '<think>' + after_result
        if last_result_idx != -1:
            new_trajectory = new_trajectory[:last_result_idx + len('</result>')] + after_result
        else:
            new_trajectory = after_result
        new_trajectory += hint_content
        python_times, search_times = calculate_python_times(new_trajectory), calculate_search_times(new_trajectory)
        direct_generate_input = process_modify_previous_trajectory_input(self.tokenizer_m1, self.sample_stat["input"], new_trajectory)
        new_trajectory, logs, _ = await self.generate_initial_trajectory(
            trajectory=new_trajectory,
            python_times=python_times,
            search_times=search_times,
            max_python_times=self.args.max_python_times+self.args.additional_python_times,
            max_search_times=self.args.max_search_times+self.args.additional_search_times,
            input=direct_generate_input,
        )
        return new_trajectory

    async def evolve_incorrect_trajectory_level(self, trajectory):
        # 对于整个轨迹做修正
        new_evolved_trajectory = await self.evolve_incorrect_trajectory_hint_trajectory(trajectory)
        return [new_evolved_trajectory]

    def select_best_trajectory(self, prediction_results, golden_answers, have_found_idxs=[0]):
        buffer_trajectories = self.sample_stat["evolve_trajectories_buffer"]
        source = self.sample_stat["source"]

        # 记录每个轨迹的原始下标
        original_indices = list(range(len(buffer_trajectories)))

        # 计算排序依据
        if source in ['openr1', 'numina-tir', 'numina-cot', 'aime']:
            # 先按prediction_results为True的排前面
            sort_keys = []
            for i, res in enumerate(prediction_results):
                # True的排前面，False的排后面
                sort_keys.append((not res, 0))  # 先按True/False, 工具调用次数后面再补
        else:
            # 先按F1分数从高到低
            f1_scores = []
            for i in range(len(buffer_trajectories)):
                if not golden_answers:
                    f1_scores.append(0.0)
                else:
                    # 计算每个golden_answer的F1，取最大
                    item_f1_scores = [calculate_f1_score(prediction_results[i], golden_answers[j]) for j in range(len(golden_answers))]
                    f1_scores.append(max(item_f1_scores))
            # 排序key为(-f1, 0)，负号表示降序
            sort_keys = [(-f1, 0) for f1 in f1_scores]

        # 计算工具调用次数
        tool_call_counts = [
            count_valid_tags(buffer_trajectories[i], "python") + count_valid_tags(buffer_trajectories[i], "search")
            for i in range(len(buffer_trajectories))
        ]

        # 更新排序key，第二项为工具调用次数
        for i in range(len(sort_keys)):
            sort_keys[i] = (sort_keys[i][0], tool_call_counts[i])

        # 排序，记录新顺序下的原始下标
        sorted_items = sorted(
            zip(buffer_trajectories, original_indices, sort_keys),
            key=lambda x: (x[2][0], x[2][1])
        )
        sorted_indices = [item[1] for item in sorted_items]

        # 找到第一个不在have_found_idxs中的轨迹
        now_best_idx = None
        for idx in sorted_indices:
            if idx not in have_found_idxs:
                now_best_idx = idx
                break
        if now_best_idx is None:
            # 如果都在have_found_idxs中，返回第一个
            now_best_idx = sorted_indices[0]
        if self.use_log:
            print('Now Best Idx: ================================')
            print(f'Current Question: {self.sample_stat["input"]}')
            print(now_best_idx)
            print('===============================================')
        best_trajectory = buffer_trajectories[now_best_idx]
        # 更新have_found_idxs
        new_have_found_idxs = have_found_idxs.copy()
        if now_best_idx not in new_have_found_idxs:
            new_have_found_idxs.append(now_best_idx)
        best_prediction_result = prediction_results[now_best_idx] if now_best_idx < len(prediction_results) else None
        return best_trajectory, new_have_found_idxs, best_prediction_result

    async def evolve_trajectories_correct_incorrect(self, have_found_idxs=[0]):
        # 该方法用于根据evolve_type并发地修正轨迹
        
        # 1. 准备Golden Answers
        if isinstance(self.sample_stat["answer"], list):
            golden_answers = [str(ans) for ans in self.sample_stat["answer"]]
        else:
            golden_answers = [str(self.sample_stat["answer"])]

        # 2. 使用 select_best_trajectory 选择最佳轨迹
        if self.sample_stat["source"] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo']:
            # 数学问题，直接看llm judge结果
            best_trajectory, new_have_found_idxs, judge_result = self.select_best_trajectory(self.sample_stat["evolve_trajectories_llm_judge_results"], golden_answers, have_found_idxs)
        else:
            # qa问题，算f1分数
            best_trajectory, new_have_found_idxs, judge_result = self.select_best_trajectory(self.sample_stat["evolve_predictions"], golden_answers, have_found_idxs)

        # 3. 确定 evolve_type
        evolve_type = "incorrect"
        if self.sample_stat["source"] in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo']: # 数学问题
            evolve_type = "correct" if judge_result else "incorrect"
        else: # qa问题
            max_f1 = max([calculate_f1_score(judge_result, golden) for golden in golden_answers])
            if max_f1 > 0.8:
                evolve_type = "correct"
            else:
                evolve_type = "incorrect"

        if evolve_type == "correct":
            # 正确时需要单步修正、直接重新rollout、整条链refine

            # 首先直接rollout
            task_re_rollout = asyncio.create_task(
                self.generate_initial_trajectory()
            )

            self.sample_stat['evolve_judgements'].append('Correct, direct rollout')

            # 然后是单步修正
            evolve_trajectory_context_step_level = process_input_correct_step_refine(self.tokenizer_m3, self.sample_stat["input"], best_trajectory)
            task_step_level = asyncio.create_task(
                self.local_evolve_correct_trajectory(evolve_trajectory_context_step_level, best_trajectory)
            )

            # 然后是整条链refine
            task_refine_whole_level = asyncio.create_task(
                self.refine_trajectory_whole_level(best_trajectory)
            )

            (new_evolved_trajectory_re_rollout, _, _), new_evolved_trajectory_step_level, new_evolved_trajectory_refine_whole_level = await asyncio.gather(
                task_re_rollout, task_step_level, task_refine_whole_level
            )
            return [new_evolved_trajectory_re_rollout, new_evolved_trajectory_step_level, new_evolved_trajectory_refine_whole_level], new_have_found_idxs

        elif evolve_type == "incorrect":
            # 并发执行step level和hint level两种修正
            task_step = asyncio.create_task(
                self.evolve_incorrect_trajectory_step_level(best_trajectory)
            )
            task_trajectory = asyncio.create_task(
                self.evolve_incorrect_trajectory_level(best_trajectory)
            )
            new_evolved_trajectory_step_level, new_evolved_trajectory_level = await asyncio.gather(
                task_step, task_trajectory
            )
            return new_evolved_trajectory_step_level + new_evolved_trajectory_level, new_have_found_idxs
        else:
            raise ValueError(f"Invalid evolve type: {evolve_type}")

    async def evolve_trajectories(self):
        # 最多循环self.args.max_evolve_times次进化
        have_found_idxs = [0]
        
        for i in range(self.args.max_evolve_times):
            new_trajectories, have_found_idxs = await self.evolve_trajectories_correct_incorrect(have_found_idxs)
            
            # print('New trajectories: ', new_trajectories)
            if new_trajectories is None or len(new_trajectories) == 0 or (len(new_trajectories) == 1 and new_trajectories[0] is None):
                break # 改为break而不是return，以保持一致性
            
            # 剔除 None 或空字符串元素
            new_trajectories = [traj for traj in new_trajectories if traj is not None and traj != ""]
            
            if len(new_trajectories) == 0:
                break
            
            # print('New trajectories: ', new_trajectories)
            for new_trajectory in new_trajectories:
                self.sample_stat["evolve_trajectories_buffer"].append(new_trajectory)
                new_generated_answer = extract_answer(new_trajectory)
                self.sample_stat["evolve_predictions"].append(new_generated_answer)
                judge_result = await self.judge_correctness(question=self.sample_stat["input"], labeled_answer=self.sample_stat["answer"], pred_answer=new_generated_answer)
                if judge_result:
                    self.sample_stat["evolve_trajectories_llm_judge_results"].append(True)
                else:
                    self.sample_stat["evolve_trajectories_llm_judge_results"].append(False)

    async def run(self):
        self.sample_start_time = time.time()
        # 首先，执行第一遍初始推理，得到一条轨迹
        # await self.generate_initial_trajectory()
        initial_outputs, initial_logs, initial_prediction = await self.generate_initial_trajectory()
        # self.sample_stat["evolve_trajectories_buffer"].append(self.sample_stat["initial_output"])
        self.sample_stat["evolve_trajectories_buffer"].append(initial_outputs)
        self.sample_stat["evolve_predictions"].append(initial_prediction)

        # 之后，判断答案是否正确
        judge_result = await self.judge_correctness(question=self.sample_stat["input"], labeled_answer=self.sample_stat["answer"], pred_answer=initial_prediction)
        if judge_result:
            self.sample_stat["evolve_trajectories_llm_judge_results"].append(True)
        else:
            self.sample_stat["evolve_trajectories_llm_judge_results"].append(False)
        
        await self.evolve_trajectories()

