import imp
import sys
import os
sys.path.append(os.getcwd())
import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from prompts import *

from math_equivalence import *

from math_equivalence import _strip_string

def make_choices_format(choices):
    return '\n'.join([f'{chr(65+i)}. {c}' for i, c in enumerate(choices)])

def last_boxed_only_string(string):
    """
    提取字符串中最后一个 \boxed{} 或 \fbox{} 格式的内容
    如果找不到则返回 None
    """
    try:
        idx = string.rfind("\\boxed")
    except:
        import pdb; pdb.set_trace()
    # if "\\boxed " in string:
    #     return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string[idx:]
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_redundant_solution(solution):
    """
    对solution字符串，找到最后一个\boxed{}的内容，删除其后第一个\n及其后所有内容。
    如果找不到\boxed{}，则返回原始solution。
    """
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return solution
    # 找到\boxed{的起始位置
    i = idx
    num_left_braces_open = 0
    right_brace_idx = None
    # 向后找到匹配的右大括号
    while i < len(solution):
        if solution[i] == "{":
            num_left_braces_open += 1
        if solution[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        # 没有找到匹配的右大括号，直接返回原始solution
        return solution
    # right_brace_idx为最后一个\boxed{}的右大括号下标
    # 从right_brace_idx+1开始，查找下一个\n
    next_newline_idx = solution.find("\n", right_brace_idx + 1)
    if next_newline_idx == -1:
        # 没有\n，直接返回到right_brace_idx+1
        return solution[:right_brace_idx + 1]
    else:
        # 返回到该\n为止（包括该\n）
        return solution[:next_newline_idx + 1]

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    检查文本是否是有效的QA模板格式
    返回: (是否有效, 错误信息)
    """
    text = text.strip()
    # 检查 <think></think> 标签是否成对出现
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> tag not match"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "missing <think> or </think> tag"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> tag not match"        
    
    if not text.startswith('<think>'):
        return False, "text must start with <think>"

    if not text.endswith('</answer>'):
        return False, "text must end with </answer>"
    
    # 检查 search/result 标签的顺序
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tag not complete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tag nested order error"
            
        current_pos = result_end_pos

    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break
        
        python_end_pos = text.find('</python>', python_pos)
        result_pos = text.find('<result>', python_end_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tag not complete"

        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tag nested order error"

        current_pos = result_end_pos
    
    # 检查答案中是否包含 \boxed{} 格式
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer missing \\boxed{} format"

    # INSERT_YOUR_CODE
    # 检查<answer>和</answer>之间有且只有一个\boxed{...}
    if answer_start != -1 and answer_end != -1:
        answer_inner = text[answer_start + len('<answer>'):answer_end]
        # 检查所有\boxed{的位置
        boxed_indices = []
        idx = 0
        while True:
            idx = answer_inner.find(r'\boxed{', idx)
            if idx == -1:
                break
            boxed_indices.append(idx)
            idx += len(r'\boxed{')
        if len(boxed_indices) != 1:
            return False, f"<answer>和</answer>之间必须且只能包含一个\\boxed{{}}，当前检测到{len(boxed_indices)}个"

    # INSERT_YOUR_CODE
    # 检查<answer> </answer>之间不能有任何<think> </think> <python> </python> <search> </search>
    if answer_start != -1 and answer_end != -1:
        answer_inner = text[answer_start + len('<answer>'):answer_end]
        forbidden_tags = ['<think>', '</think>', '<python>', '</python>', '<search>', '</search>']
        for tag in forbidden_tags:
            if tag in answer_inner:
                return False, f"<answer>和</answer>之间不能包含{tag}标签"
    
    return True, "format correct"

def extract_answer(full_text: str, prompt: str = "") -> str:
    if prompt:
        text = full_text[len(prompt) :]
    else:
        text = full_text
    last_answer_end = text.rfind("</answer>")
    if last_answer_end != -1:
        temp_text = text[:last_answer_end]
        last_answer_start = temp_text.rfind("<answer>")
        if last_answer_start != -1:
            temp_answer = text[last_answer_start + len("<answer>") : last_answer_end]
        else:
            temp_answer = None
    else:
        temp_answer = None

    if temp_answer:
        boxed_answer = temp_answer.strip()
        boxed_answer = last_boxed_only_string(boxed_answer)
        if (
            boxed_answer
            and boxed_answer.startswith("\\boxed{")
            and boxed_answer.endswith("}")
        ):
            boxed_content = boxed_answer[7:-1]
            boxed_answer = boxed_content
            if (
                boxed_answer
                and boxed_answer.startswith("\\text{")
                and boxed_answer.endswith("}")
            ):
                boxed_content = boxed_answer[6:-1]
                boxed_answer = boxed_content

        if not boxed_answer:
            final_answer = temp_answer
        else:
            final_answer = boxed_answer
    else:
        boxed_answer = text.strip()
        final_answer = last_boxed_only_string(boxed_answer)
        if (
            final_answer
            and final_answer.startswith("\\boxed{")
            and final_answer.endswith("}")
        ):
            final_answer = final_answer[7:-1]

            if (
                final_answer
                and final_answer.startswith("\\text{")
                and final_answer.endswith("}")
            ):
                final_answer = final_answer[6:-1]
    return final_answer

def calculate_f1_score(prediction: str, reference: str) -> float:
    if prediction is None:
        prediction = ''
    prediction_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    return 2 * num_same / (len(prediction_tokens) + len(reference_tokens)) if len(prediction_tokens) + len(reference_tokens) > 0 else 0

def count_valid_tags(text: str, tag: str) -> int:
    """Count valid paired tags."""
    count = 0
    current_pos = 0

    while True:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            break

        end_pos = text.find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            break

        count += 1
        current_pos = end_pos + len(end_tag)

    return count

def format_trajectory(trajectory: str, tag='python') -> str:
    """
    按照指定规则格式化trajectory，主要处理<think>、<answer>、<result>等标签。
    """

    s = trajectory

    # 1. 保证有</answer>
    if "</answer>" not in s:
        s = s.rstrip() + "</answer>"

    # 2. 保证有<answer>
    if "<answer>" not in s:
        if "\\boxed" in s:
            # 从后往前找第一个\boxed
            idx_boxed = s.rfind("\\boxed")
            # 向前找第一个\n
            idx_n = s.rfind('\n', 0, idx_boxed)
            if idx_n == -1:
                # 没有\n，直接在开头加<answer>
                return None
            else:
                s = s[:idx_n+1] + "<answer>" + s[idx_n+1:]
        else:
            # 从后往前找</answer>
            idx_ans = s.rfind("</answer>")
            if idx_ans == -1:
                return None
            else:
                # 跳过紧挨着</answer>前的\n和空格
                idx = idx_ans - 1
                while idx >= 0 and s[idx] in ['\n', ' ']:
                    idx -= 1
                idx_n = s.rfind('\n', 0, idx+1)
                if idx_n == -1:
                    return None
                else:
                    s = s[:idx_n+1] + "<answer>" + s[idx_n+1:]

    # 3. 保证以<think>开头
    s_strip = s.lstrip()
    if not s_strip.startswith("<think>"):
        s = "<think>" + s

    # 4. 在<answer>前插入</think>
    idx_ans = s.find("<answer>")
    if idx_ans != -1:
        # 找<answer>前第一个不是\n或空格的地方
        idx = idx_ans - 1
        while idx >= 0 and s[idx] in ['\n', ' ']:
            idx -= 1
        insert_pos = idx + 1
        s = s[:insert_pos] + "</think>" + s[insert_pos:]

    # 5. 在所有<tag>前插入</think>
    tag_open = f"<{tag}>"
    tag_positions = []
    start = 0
    while True:
        idx_tag = s.find(tag_open, start)
        if idx_tag == -1:
            break
        tag_positions.append(idx_tag)
        start = idx_tag + len(tag_open)
    # 逆序插入，避免位置错乱
    for idx_tag in reversed(tag_positions):
        idx = idx_tag - 1
        while idx >= 0 and s[idx] in ['\n', ' ']:
            idx -= 1
        insert_pos = idx + 1
        s = s[:insert_pos] + "</think>" + s[insert_pos:]

    # 6. 在所有</result>后插入<think>
    result_close = "</result>"
    result_positions = []
    start = 0
    while True:
        idx_res = s.find(result_close, start)
        if idx_res == -1:
            break
        result_positions.append(idx_res)
        start = idx_res + len(result_close)
    # 正序插入
    offset = 0
    for idx_res in result_positions:
        idx_res += offset
        after = idx_res + len(result_close)
        # 跳过后面所有的\n和空格
        idx = after
        while idx < len(s) and s[idx] in ['\n', ' ']:
            idx += 1
        s = s[:idx] + "<think>" + s[idx:]
        offset += len("<think>")

    # 处理所有<think></think>对，若中间为空或仅有\n和空格，则整体删除
    start = 0
    while True:
        idx_think_open = s.find("<think>", start)
        if idx_think_open == -1:
            break
        idx_think_close = s.find("</think>", idx_think_open + len("<think>"))
        if idx_think_close == -1:
            break
        # 获取<think>和</think>之间的内容
        content_between = s[idx_think_open + len("<think>"):idx_think_close]
        if content_between == "" or all(c in ['\n', ' '] for c in content_between):
            # 删除整个<think>...</think>
            s = s[:idx_think_open] + s[idx_think_close + len("</think>"):]
            # 删除后，start不变，继续查找当前位置
        else:
            # 若内容不全是空白，则从</think>后继续查找
            start = idx_think_close + len("</think>")
    # 找s中所有可能的“<think></think>”子段，并删去
    while True:
        idx_think_open = s.find("<think></think>")
        if idx_think_open == -1:
            break
        s = s[:idx_think_open] + s[idx_think_open + len("<think></think>"):]

    return s

def remove_result_tags(s):
    # 用find方法去除所有<result>...</result>内容
    result_start = s.find('<result>')
    while result_start != -1:
        result_end = s.find('</result>', result_start)
        if result_end == -1:
            break
        s = s[:result_start] + s[result_end + len('</result>'):]
        result_start = s.find('<result>')
    return s

def extract_results(s):
    """
    提取s中所有的<result>...</result>中间的内容（倘若<result></result>这种形式，那么也要提取一个""出来）
    返回list[str]
    """
    results = []
    result_open = "<result>"
    result_close = "</result>"
    pos = 0
    while True:
        idx_start = s.find(result_open, pos)
        if idx_start == -1:
            break
        idx_start_content = idx_start + len(result_open)
        idx_end = s.find(result_close, idx_start_content)
        if idx_end == -1:
            break
        content = s[idx_start_content:idx_end]
        results.append(content)
        pos = idx_end + len(result_close)
    return results

def check_result_correctness(results):
    # INSERT_YOUR_CODE
    """
    如果任何一个result中，有”Execution Timeout“字样，或者”Tool execute failed“字样，
    或者"error" in result.lower()，或者result为空，都认为这个results是错误的
    """
    for result in results:
        if (
            result == "" or
            "Execution Timeout" in result or
            "Tool execute failed" in result or
            "error" in result.lower()
        ):
            return False
    return True

def process_hint_position(output: str, hint: str) -> str:
    # 找到 hint 的位置
    idx = output.find(hint)
    if idx == -1:
        return output  # 没找到 hint，不做处理

    # 从 hint 位置往前找到最近的 <think>
    think_pos = output.rfind("<think>", 0, idx)
    if think_pos == -1:
        return output  # 没找到 <think>，不处理

    # 找到 <think> 前面第一个非空白字符
    i = think_pos - 1
    while i >= 0 and output[i] in [' ', '\n']:
        i -= 1
    before_think = output[:i+1] if i >= 0 else ""

    # 检查是否以 </result> 结尾
    ends_with_result = before_think.endswith("</result>")

    # 如果不是以 </result> 结尾，就要删除从连续的 <think> 开始的所有
    if not ends_with_result:
        start = think_pos
        # 向前找连续的 <think>
        while True:
            prev_think = output.rfind("<think>", 0, start)
            if prev_think == -1:
                break
            between = output[prev_think + len("<think>"):start]
            if between.strip():  # 中间有非空白内容，说明不连续
                break
            start = prev_think
        # 删除这段连续的 <think>
        output = output[:start] + output[think_pos + len("<think>"):]
        idx = output.find(hint)  # 更新 hint 位置，因为字符串改变了

    # 现在检查 hint 前面第一个非空白字符是否以 </result> 结尾
    j = idx - 1
    while j >= 0 and output[j] in [' ', '\n']:
        j -= 1
    before_hint = output[:j+1] if j >= 0 else ""
    ends_with_result_before_hint = before_hint.endswith("</result>")

    # 如果以 </result> 结尾，则在其后添加一个 <think>
    if ends_with_result_before_hint:
        insert_pos = len(before_hint)
        output = output[:insert_pos] + "<think>" + output[insert_pos:]

    return output

def check_answer_correctness(golden_answers, predictions, type='math'):
    results = []
    if type == 'math':
        for prediction in predictions:
            if is_equiv(prediction, golden_answers):
                results.append(1)
            else:
                results.append(0)
    elif type == 'qa':
        if not isinstance(golden_answers, list):
            golden_answers = [golden_answers]
        for prediction in predictions:
            # if max([calculate_f1_score(prediction, golden_answer) for golden_answer in golden_answers]) > 0.8:
            #     results.append(True)
            # else:
            #     results.append(False)
            results.append(max([calculate_f1_score(prediction, golden_answer) for golden_answer in golden_answers]))
    else:
        raise ValueError(f"Invalid type: {type}")
    return results

def check_answer_correctness_deepsearch(golden_answers, predictions):
    results = []
    for prediction in predictions:
        results.append(
            max(
                [
                    calculate_f1_score(_strip_string(prediction, deepsearch=True), _strip_string(golden_answer, deepsearch=True))
                    for golden_answer in golden_answers
                ]
            )
        )
    results = [1 if result >= 0.75 else 0 for result in results]
    return results