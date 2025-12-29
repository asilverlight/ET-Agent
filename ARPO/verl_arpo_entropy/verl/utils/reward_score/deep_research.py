import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
# from validate_format import *
from .validate_format import *


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.
    
    Args:
        string: String to extract \boxed{} from
        
    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
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

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval





def compute_score(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    valid_template, reason = validate_format(response)
    
    if not valid_template:
        print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        return result
    
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    # Bonus for using multiple tools correctly
    if f1_score > 0 and "</search>" in response and "</python>" in response:
        print(f"--------------------------------correct answer with multi tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("{path_of_your_model}/Qwen2.5-1.5B-Instruct")
    response = "<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To solve the given problem, we will use both natural language reasoning and Python code. Let's break down the problem step-by-step.\n\n1. **Given Equations:**\n   \\[\n   \\log_{10}(\\sin x) + \\log_{10}(\\cos x) = -1\n   \\]\n   Using the properties of logarithms, we can combine the logs:\n   \\[\n   \\log_{10}(\\sin x \\cos x) = -1 \\implies \\sin x \\cos x = 10^{-1} = \\frac{1}{10}\n   \\]\n   Recall the double-angle identity for sine: \\(\\sin(2x) = 2 \\sin x \\cos x\\). So,\n   \\[\n   \\sin(2x) = 2 \\left(\\frac{1}{10}\\right) = \\frac{1}{5}\n   \\]\n\n2. **Second Given Equation:**\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}(\\log_{10}(n) - 1)\n   \\]\n   We can rewrite the right-hand side using properties of logarithms:\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}\\log_{10}(n) - \\frac{1}{2} = \\log_{10}(n^{1\/2}) - \\log_{10}(10^{1\/2}) = \\log_{10}\\left(\\frac{\\sqrt{n}}{\\sqrt{10}}\\right)\n   \\]\n   This implies:\n   \\[\n   \\sin x + \\cos x = \\frac{\\sqrt{n}}{\\sqrt{10}}\n   \\]\n   Squaring both sides, we get:\n   \\[\n   (\\sin x + \\cos x)^2 = \\frac{n}{10}\n   \\]\n   Expanding the left side:\n   \\[\n   \\sin^2 x + \\cos^2 x + 2 \\sin x \\cos x = \\frac{n}{10}\n   \\]\n   Using the Pythagorean identity \\(\\sin^2 x + \\cos^2 x = 1\\) and the earlier result \\(\\sin x \\cos x = \\frac{1}{10}\\):\n   \\[\n   1 + 2 \\left(\\frac{1}{10}\\right) = \\frac{n}{10} \\implies 1 + \\frac{1}{5} = \\frac{n}{10} \\implies \\frac{6}{5} = \\frac{n}{10} \\implies n = 12\n   \\]\n\nNow, let's verify this with Python code to ensure the result is accurate.\n </think><python>\nimport math\r\nfrom sympy import symbols, Eq, solve, log, sin, cos\r\n\r\n# Define the variable\r\nx = symbols('x')\r\n\r\n# First equation: log10(sin(x)) + log10(cos(x)) = -1\r\neq1 = Eq(log(sin(x), 10) + log(cos(x), 10), -1)\r\n\r\n# Solve for sin(x) * cos(x)\r\nsin_x_cos_x = solve(eq1, sin(x) * cos(x))[0]\r\nsin_x_cos_x = sin_x_cos_x.simplify()\r\n\r\n# We know sin(x) * cos(x) = 1\/10\r\nsin_x_cos_x_value = 1 \/ 10\r\n\r\n# Second equation: log10(sin(x) + cos(x)) = 1\/2 * (log10(n) - 1)\r\nn = symbols('n')\r\neq2 = Eq(log(sin(x) + cos(x), 10), (1\/2) * (log(n, 10) - 1))\r\n\r\n# We know (sin(x) + cos(x))^2 = 1 + 2 * sin(x) * cos(x) = 1 + 2 * (1\/10) = 6\/5\r\nsin_x_plus_cos_x_squared = 6 \/ 5\r\nsin_x_plus_cos_x = math.sqrt(sin_x_plus_cos_x_squared)\r\n\r\n# Substitute sin(x) + cos(x) into the second equation\r\nlhs = log(sin_x_plus_cos_x, 10)\r\nrhs = (1\/2) * (log(n, 10) - 1)\r\n\r\n# Solve for n\r\nn_value = solve(Eq(lhs, rhs), n)[0]\r\nprint(n_value)\n</python><result>\n\n12.0000000000000\n</result><search> query </search><result> result </result><answer>The value of \\( n \\) is \\(\\boxed{12}\\). The calculations and the verification using Python both confirm that the final answer is indeed \\( n = 12 \\).</answer><|im_end|>"

    answer = "12"
    extra_info = {"tokenizer": tokenizer}
    res = compute_score("test_data_source", response, answer, extra_info)
    print(res)
