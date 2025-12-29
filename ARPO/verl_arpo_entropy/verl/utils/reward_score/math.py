# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


# def compute_score(solution_str, ground_truth) -> float:
#     retval = 0.0
#     if isinstance(ground_truth, list):
#         ground_truth = ground_truth[0]
#     try:
#         string_in_last_boxed = last_boxed_only_string(solution_str)
#         if string_in_last_boxed is not None:
#             answer = remove_boxed(string_in_last_boxed)
#             if is_equiv(answer, ground_truth):
#                 retval = 1.0
#     except Exception as e:
#         print(e)

#     return retval

# from validate_format import *
# from .validate_format import *
try:
    from .validate_format import *
except Exception:
    from verl.utils.reward_score.validate_format import *
from typing import Optional, Tuple
import re

def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check the order of python/result
    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break

        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</python>', python_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"

def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from the text within <answer> tags.
    
    Args:
        text: The text to extract answer from
        
    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)

def compute_score(solution_str, ground_truth, extra_info=None) -> float:
    result = {
        "score": -1,
        "reason": "",
        "answer": "",
        "f1_score": -1
    }
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    
    response = solution_str
    valid_template, reason = validate_format(response)
    if not valid_template:
        print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["reason"] = reason
        return result

    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["reason"] = "cannot extract answer"
        return result

    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            # answer = remove_boxed(string_in_last_boxed)
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


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
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

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def remove_right_units_deepsearch(string):
    """将所有 \\text{...} 的外层去掉，仅保留其内部文本内容。
    支持多处出现与嵌套花括号的匹配；若遇到不完整的花括号结构则尽量保留原文。
    例如：
      - "20 \\text{ years and } 81 \\text{ days}, 13.30" -> "20 years and 81 days, 13.30"
      - "17 \\text{ months and } 50\\%" -> "17 months and 50\\%"
    """
    i = 0
    n = len(string)
    out = []
    # has_text = False
    # if '\\text' in string:
    #     has_text = True
    # if has_text:
    #     print('================ before remove_right_units_deepsearch: ', string)
    while i < n:
        if string.startswith("\\text", i):
            j = i + 5  # 跳过 \\text
            # 可选空白
            while j < n and string[j].isspace():
                j += 1
            if j < n and string[j] == '{':
                j += 1
                depth = 1
                content = []
                while j < n and depth > 0:
                    ch = string[j]
                    if ch == '{':
                        depth += 1
                        content.append(ch)
                        j += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            j += 1  # 跳过与之匹配的右花括号
                            break
                        else:
                            content.append(ch)
                            j += 1
                    else:
                        content.append(ch)
                        j += 1
                if depth == 0:
                    out.append(''.join(content))
                    i = j
                    continue
                else:
                    # 未能正确闭合，保留原文片段，避免误删
                    out.append(string[i:j])
                    i = j
                    continue
            else:
                # \\text 后未跟 {，视为普通文本
                out.append("\\text")
                i = i + 5
                continue
        else:
            out.append(string[i])
            i += 1
    result = ''.join(out)
    # 轻度空白规范化：折叠连续空格/Tab，去掉首尾空白，避免因去壳导致的双空格
    result = re.sub(r'[ \t]+', ' ', result).strip()
    # if has_text:
    #     print('================ after remove_right_units_deepsearch: ', result)
    return result


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string, deepsearch=False):
    # linebreaks
    if not isinstance(string, str):
        string = str(string)
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    # string = remove_right_units(string)
    if deepsearch:
        string = remove_right_units_deepsearch(string)
    else:
        string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
