import re
import sys
import os
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


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


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
