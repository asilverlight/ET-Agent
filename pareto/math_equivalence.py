import sys
import os
import re
sys.path.append(os.getcwd())

def _fix_fracs(string):
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
                except:
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

def _fix_a_slash_b(string):
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
    except:
        return string

def _remove_right_units(string):
    """去掉右侧单位/说明性的 \text{...}，并兼容多个出现的情况。
    策略：若存在任意 \text{...}，截取第一个 \text{ 之前的内容；否则原样返回。
    这样示例：
      - "20 \\text{ years and } 81 \\text{ days}, 13.30" -> "20"
      - "17 \\text{ months and } 50\\% " -> "17"
    """
    # 寻找第一个 \text{ 的位置（允许 \text 与 { 之间有可选空格）
    match = re.search(r"\\text\s*\{", string)
    if match:
        # 截断到 \text 出现之前，同时去掉右侧多余的空格和常见分隔符
        prefix = string[: match.start()]
        prefix = prefix.rstrip(" \t,;:")
        return prefix
    return string

def _remove_right_units_deepsearch(string):
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


def _fix_sqrt(string):
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

# def remove_boxed(s):
#     if "\\boxed" in s:
#         left = "\\boxed"
#         assert s[: len(left)] == left
#         return s[len(left) :]
#     left = "\\boxed{"
#     assert s[: len(left)] == left
#     assert s[-1] == "}"
#     return s[len(left) : -1]

def _strip_string(string, deepsearch=False):
    if not isinstance(string, str):
        string = str(string)
    string = string.strip()
    if string.startswith("\\boxed") and string.endswith("}"):
        string = string[7:-1]
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    if deepsearch:
        string = _remove_right_units_deepsearch(string)
    else:
        string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2