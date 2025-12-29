import time
import math
from .utils import *
from typing import List
import random

def format_trajectory_steps(trajectory=None):
    """
    将trajectory字符串按照每个</result>为分界点切分为步骤，每个</result>归到其前面的步骤中。
    返回步骤列表。
    """
    if trajectory is None:
        return ""
    steps = ""
    last_idx = 0
    step_num = 1
    import re
    for match in re.finditer(r"</result>", trajectory):
        end_idx = match.end()
        step = trajectory[last_idx:end_idx].strip()
        # 加上序号
        steps += f"Step {step_num}: {step}\n"
        last_idx = end_idx
        step_num += 1
    # 如果最后还有残留内容（比如最后是<answer>），也加进去
    if last_idx < len(trajectory):
        rest = trajectory[last_idx:].strip()
        if rest:
            steps += f"Step {step_num}: {rest}\n"
    return steps

def log_trajectory_output(tool_result, outputs, logs, trajectory_context):
    outputs += tool_result
    logs.append(tool_result)
    trajectory_context += tool_result
    return outputs, logs, trajectory_context

def log_modify_wrong_trajectory_output(*args, content, **kwargs):
    # 如果没有位置参数，则从关键字参数中取
    if len(args) == 0:
        modify_wrong_trajectory_context = kwargs.get("modify_wrong_trajectory_context")
        currect_trajectory = kwargs.get("currect_trajectory")
        logs = kwargs.get("existing_logs")
    else:
        modify_wrong_trajectory_context, currect_trajectory, logs = args
    content_to_log = kwargs.get("tool_result", content)
    modify_wrong_trajectory_context += content_to_log
    currect_trajectory += content_to_log
    logs.append(content_to_log)
    return modify_wrong_trajectory_context, currect_trajectory, logs

async def call_python(tool_executor, python_code, outputs='', logs=[], trajectory_context='', use_log=False, max_python_result_length=1200):
    python_result = await tool_executor.execute(
        "python", python_code, timeout=30
    )
    if len(python_result) > max_python_result_length:
        python_result = python_result[:max_python_result_length] + '...'
    tool_result = f"<result>{python_result}</result>"
    if use_log:
        print('Current Python Result: ================================')
        print(tool_result)
        print('===============================================')
    # return log_func(*args, content=tool_result)
    return tool_result, log_trajectory_output(tool_result, outputs, logs, trajectory_context)

async def call_search(tool_executor, search_query, outputs='', logs=[], trajectory_context='', source='', use_log=False, compatible_search=False, use_local_search=False):
    if not compatible_search:
        if use_local_search:
            search_result = await tool_executor.execute(
                "localsearch", search_query, timeout=60, existing_logs=logs
            )
        else:
            search_result = await tool_executor.execute(
                "websearch", search_query, timeout=60, existing_logs=logs
            )
    else:
        if source in ['qa']:
            search_result = await tool_executor.execute(
                "localsearch", search_query, timeout=60, existing_logs=logs
            )
        else:
            search_result = await tool_executor.execute(
                "websearch", search_query, timeout=60, existing_logs=logs
            )
    if search_query is None or search_result is None:
        tool_result = f"<result></result>"
    else:
        tool_result = f"<result>{search_result}</result>"
    if use_log:
        print('Current Search Result: ================================')
        print(tool_result)
        print('===============================================')
    return tool_result, log_trajectory_output(tool_result, outputs, logs, trajectory_context)

async def call_local_llm_for_generate_trajectory(vllm_pool, in_context, stop, sampling_params, sampling_params_nostop, session_id, use_log=False):
    try_time = 0
    while try_time < 4:
        try_time += 1
        # print('sampling Params: ================================')
        # print(sampling_params)
        # print('===============================================')
        result = await vllm_pool.generate(
            in_context,
            (
                sampling_params
                if stop is True
                else sampling_params_nostop
            ),
            session_id=session_id
        )
        if not result:
            continue
        output = result.choices[0].text.strip()
        # output = output.split("<result>")[0]
        if "</answer>" in output:
            output = output.split("</answer>")[0] + "</answer>"
        if "</search>" in output:
            output = output.split("</search>")[0] + "</search>"
        if "</python>" in output:
            output = output.split("</python>")[0] + "</python>"
        if use_log:
            print('Current Output: ================================')
            print(output)
            print('===============================================')
        return output
    return ''

async def call_llm_for_generate_trajectory(vllm_pool, in_context, stop, sampling_params, sampling_params_nostop, session_id, outputs, logs):
    output = await call_local_llm_for_generate_trajectory(vllm_pool, in_context, stop, sampling_params, sampling_params_nostop, session_id)
    outputs, logs, in_context = log_trajectory_output(output, outputs, logs, in_context)
    return output, outputs, logs, in_context

async def call_local_llm_for_judge_correctness(vllm_pool, sampling_params, input, session_id):
    output = ''
    try_time = 0
    while try_time < 4:
        try_time += 1
        result = await vllm_pool.generate(
            input, sampling_params, session_id
        )
        if not result:
            return False
        output = result.choices[0].text.lower().strip()
        if 'correct' in output and 'incorrect' not in output:
            return True
    return False

async def call_local_llm_for_naive_generation(vllm_pool, sampling_params, input, session_id):
    output = ''
    try_time = 0
    while try_time < 4:
        try_time += 1
        result = await vllm_pool.generate(input, sampling_params, session_id)
        if not result:
            continue
        output = result.choices[0].text.lower().strip()
        return output
    return None

async def call_local_llm_for_evolve_correct_trajectory(vllm_pool, sampling_params, input, session_id):
    # 用于首先生成进化的策略
    output = ''
    try_time = 0
    while try_time < 4:
        try_time += 1
        result = await vllm_pool.generate(input, sampling_params, session_id)
        if not result:
            continue
        output = result.choices[0].text.lower().strip()
        return output
    return None

    