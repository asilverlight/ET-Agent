import re
import sys
import os
import math
from collections import Counter
from copy import deepcopy

sys.path.append(os.getcwd())


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]

def split_trajectory_to_steps(trajectory):
    # 用正则分割，保留</result>在每一段末尾
    pattern = re.compile(r'(.*?</result>)', re.DOTALL)
    steps = pattern.findall(trajectory)
    # 如果最后还有残留内容（如<answer>），也加进去
    rest = pattern.sub('', trajectory)
    if rest.strip():
        steps.append(rest)
    # 拼接成 step i: 第i段
    result = ""
    for i, step in enumerate(steps):
        result += f"#### Step {i + 1}: {step.strip()}\n"
    return result.strip()

def calculate_python_times(trajectory):
    python_times = 0
    start = 0
    while True:
        start_tag = trajectory.find("<python>", start)
        if start_tag == -1:
            break
        end_tag = trajectory.find("</python>", start_tag + len("<python>"))
        if end_tag == -1:
            break
        python_times += 1
        start = end_tag + len("</python>")
    return python_times

def calculate_search_times(trajectory):
    search_times = 0
    start = 0
    while True:
        start_tag = trajectory.find("<search>", start)
        if start_tag == -1:
            break
        end_tag = trajectory.find("</search>", start_tag + len("<search>"))
        if end_tag == -1:
            break
        search_times += 1
        start = end_tag + len("</search>")
    return search_times

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def split_trajectory_to_logs(trajectory):
    """
    首先，以</result>为划分，将trajectory划分为若干段，每个</result>并入其前面的那段。
    之后，对于每一段，从后往前找到遇到的第一个</search>或</python>，再以此为划分依据，将每小段trajectory再次划分
    返回：所有小段组成的list
    """
    logs = []
    # 1. 以</result>为分割点，每个</result>并入前一段
    segments = []
    last = 0
    for match in re.finditer(r"</result>", trajectory):
        end = match.end()
        segments.append(trajectory[last:end])
        last = end
    # 如果最后还有剩余内容
    if last < len(trajectory):
        segments.append(trajectory[last:])

    # 2. 对每一段，从后往前找到第一个</search>或</python>，再分割
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 从后往前找</search>或</python>
        idx_search = seg.rfind("</search>")
        idx_python = seg.rfind("</python>")
        # 取最大（即最靠后的）
        idx = max(idx_search, idx_python)
        if idx != -1:
            # 包含</search>或</python>的部分
            part1 = seg[: idx + len("</search>") if idx == idx_search else idx + len("</python>")]
            part2 = seg[idx + len("</search>") if idx == idx_search else idx + len("</python>") :]
            if part1.strip():
                logs.append(part1.strip())
            if part2.strip():
                logs.append(part2.strip())
        else:
            logs.append(seg)
    return logs

def last_boxed_only_string(string) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return ""
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
        retval = string[idx : right_brace_idx + 1]
    return retval


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


def transfer_claude_input_format(messages):
    if messages[0]["role"] == "system":
        out_system_prompt = [{"text": messages[0]["content"]}]
        messages = messages[1:]
    else:
        out_system_prompt = [{"text": "You are a helpful assistant. "}]

    new_message_format = {"role": "", "content": [{"text": ""}]}

    out_messages = []
    for message in messages:
        new_message = deepcopy(new_message_format)
        new_message["role"] = message["role"]
        new_message["content"][0]["text"] = message["content"]
        out_messages.append(new_message)

    return out_system_prompt, out_messages

def make_choices_format(choices):
    return '\n'.join([f'{chr(65+i)}. {c}' for i, c in enumerate(choices)])

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    检查文本是否是有效的QA模板格式
    返回: (是否有效, 错误信息)
    """
    # 检查 <think></think> 标签是否成对出现
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> tag not match"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "missing <think> or </think> tag"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> tag not match"        
    
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
    
    return True, "format correct"

def calculate_f1_score(prediction: str, reference: str) -> float:
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

def calculate_sigmoid(x):
    return 1 / (1 + math.exp(-x))