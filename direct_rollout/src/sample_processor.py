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
        vllm_pool,
        tokenizer_infer,
        args,
        sample_stat,
        session_id,
    ):
        self.prompt_manager = prompt_manager
        self.tool_executor = tool_executor
        self.vllm_pool = vllm_pool
        self.tokenizer_infer = tokenizer_infer
        self.args = args
        self.use_log = args.use_log
        self.compatible_search = args.compatible_search
        self.use_local_search = args.use_local_search
        self.max_rounds = args.max_rounds
        self.max_search_result_length = args.max_search_result_length
        self.max_python_result_length = args.max_python_result_length
        self.sample_stat = sample_stat
        self.question = sample_stat.get("question", sample_stat["input"])
        # system_prompt = self.prompt_manager.get_system_prompt(self.format)

        if not session_id:
            session_content = f"{ToolStar_Prompt}_{self.question}"
            session_id = hashlib.md5(session_content.encode()).hexdigest()
            # print('Session ID: ================================')
            # print(session_id)
            # print('===============================================')
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

    async def generate_initial_trajectory(self, trajectory=None, python_times=0, search_times=0, max_python_times=None, max_search_times=None, input=None): # 做最原始的toolstar推理
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
            generate_initial_trajectory_context = process_generate_initial_trajectory_input(self.tokenizer_infer, self.sample_stat["input"])
        while True:
            if self.use_log:
                print('Current Prompt: ================================')
                print(generate_initial_trajectory_context)
                print('===============================================')
            output, outputs, logs, generate_initial_trajectory_context = await call_llm_for_generate_trajectory(self.vllm_pool, generate_initial_trajectory_context, True, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, outputs, logs)
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
                    output, outputs, logs, generate_initial_trajectory_context = await call_llm_for_generate_trajectory(self.vllm_pool, generate_initial_trajectory_context, False, self.args.sampling_params, self.args.sampling_params_nostop, self.session_id, outputs, logs)
                    if not output:
                        break
                else:
                    break
        prediction = extract_answer(outputs)
        return outputs, prediction, search_times, python_times

    async def run(self):
        self.sample_start_time = time.time()
        # 采用并发方式，运行 self.max_rounds 次 generate_initial_trajectory()
        tasks = []
        for _ in range(self.max_rounds):
            tasks.append(asyncio.create_task(self.generate_initial_trajectory()))
        results = await asyncio.gather(*tasks)

        for (outputs, prediction, search_times, python_times) in results:
            self.sample_stat["rollout_outputs"].append(outputs)
            self.sample_stat["predictions"].append(prediction)
            self.sample_stat["tool_counts"].append(
                {
                    'search_times': search_times,
                    'python_times': python_times,
                    'total_times': search_times + python_times,
                }
            )
