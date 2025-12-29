# from .validate_format import *
# from .math import *
try:
    from .validate_format import *
    from .math import *
except Exception:
    from verl.utils.reward_score.validate_format import *
    from verl.utils.reward_score.math import *

# 显式导入标准库 math，避免与本地 .math 混淆
import math as _stdlib_math
from typing import Any, Dict, Optional

def compute_score_math(solution_str, ground_truth, extra_info=None) -> float:
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    
    response = solution_str
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = "cannot extract answer"
        return result

    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            if string_in_last_boxed.startswith('\\boxed') and string_in_last_boxed.endswith('}'):
                answer = string_in_last_boxed[7:-1]
            else:
                answer = string_in_last_boxed
            if is_equiv(answer, ground_truth):
                result["score"] = 1.0
                result["answer"] = answer
                result["f1_score"] = get_f1_score(answer, ground_truth)
            else:
                result["score"] = 0.0
                result["reason"] = "answer is not equivalent to ground truth"
                result["answer"] = answer
                result["f1_score"] = get_f1_score(answer, ground_truth)
    except Exception as e:
        print(e)
    return result

def compute_score_deep_research(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        
    Returns:
        Dict[str, Any]: A dictionary containing the score and additional information
    """
    
    # 初始化统一的返回结构
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    
    response = solution_str
    
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        temp_boxed = last_boxed_only_string(answer_part)
        if temp_boxed.startswith('\\boxed') and temp_boxed.endswith('}'):
            answer = temp_boxed[7:-1]
        else:
            answer = temp_boxed
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = f"find box error: {e}"
        return result
    
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score

    result["score"] = f1_score
    
    return result


def compute_score(
    data_source, 
    solution_str, 
    ground_truth, 
    extra_info, 
    current_response_length=None, 
    other_solutions=None, 
    other_response_lengths=None, 
    sigma_tool=0.2, 
    sigma_length=0.5
    ):
    """
    Compute reward score for a solution based on the ground truth.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        current_response_length: The length of the current response
        other_solutions: A list of other solutions
        other_response_lengths: A list of other response lengths
    """
    if data_source == "math_effi_rl":
        compute_score_func = compute_score_math
    elif data_source == "qa_effi_rl":
        compute_score_func = compute_score_deep_research
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source}")

    # print(sigma_tool)
    # print(sigma_length)
    # assert False

    # 首先，计算当前solution和平均solution的工具调用次数差异，以及平均长度差异
    solutions = [solution_str] + other_solutions
    tool_calls = [count_valid_tags(solution, 'search') + count_valid_tags(solution, 'python') for solution in solutions]
    average_tool_calls = sum(tool_calls) / len(tool_calls)

    response_lengths = [current_response_length] + other_response_lengths
    # 对 response_lengths 做 z-score 正态化，并将 current_response_length 放缩到归一化后的位置
    mean_len = sum(response_lengths) / len(response_lengths)
    variance = sum((x - mean_len) ** 2 for x in response_lengths) / len(response_lengths)
    std_len = _stdlib_math.sqrt(variance)
    if std_len == 0:
        normalized_responses = [0.0 for _ in response_lengths]
    else:
        normalized_responses = [(x - mean_len) / std_len for x in response_lengths]
    normalized_current = normalized_responses[0]
    normalized_average = sum(normalized_responses) / len(normalized_responses)  # 理论上为 0.0

    # 直接计算正确分数（原始分，不含任何惩罚）
    base = compute_score_func(solution_str, ground_truth, extra_info)
    if isinstance(base, dict):
        raw_score = base.get("score", 0.0)
        base_answer = base.get("answer", "")
        base_f1 = base.get("f1_score", 0.0)
    else:
        raw_score = float(base)
        base_answer = ""
        base_f1 = 0.0
    # 以上得到原始正确性分数：
    # - math: 0/1（is_equiv）
    # - qa: f1 分数

    # 计算工具调用与长度惩罚（用于训练奖励，但不影响raw_score）
    tool_call = count_valid_tags(solution_str, 'search') + count_valid_tags(solution_str, 'python')
    f_tool = 2 / (1 + _stdlib_math.exp(sigma_tool * (tool_call - average_tool_calls)))

    f_length = 2 / (1 + _stdlib_math.exp(sigma_length * (normalized_current - normalized_average)))

    valid_template, reason = validate_format(solution_str)
    f_format = -1 if not valid_template else 0

    penalized_score = raw_score * f_tool * f_length + f_format

    result = {
        # 训练用奖励：包含工具与长度项（维持向后兼容）
        "score": penalized_score,
        # 额外暴露原始正确性分数，供效率指标计算与上报
        "raw_score": raw_score,
        "reason": f"raw_score: {raw_score}, f_tool: {f_tool}, f_length: {f_length}, f_format: {f_format}",
        "answer": base_answer,
        "f1_score": base_f1,
        "tool_call": tool_call,
    }
    # print(f'================================ tool calls =================================')
    # print(f"current_tool_call: {tool_call}")
    # print(f"average_tool_calls: {average_tool_calls}")
    # print(f"================================ response lengths =================================")
    # print(f"current_response_length: {current_response_length}")
    # print(f"normalized_current: {normalized_current}")
    # print(f"normalized_average: {normalized_average}")
    # print(f"================================ solution string =================================")
    # print(f"solution_str: {solution_str}")
    # print(f"================================ result: {result} ================================")
    return result

# def last_boxed_only_string(string):
#     idx = string.rfind("\\boxed")
#     if "\\boxed " in string:
#         return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#         if string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break
#         i += 1

#     retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

#     return retval
