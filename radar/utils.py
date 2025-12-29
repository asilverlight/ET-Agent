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

def extract_information(s):
    """
    提取s中所有的<information>...</information>中间的内容（倘若<information></information>这种形式，那么也要提取一个""出来）
    返回list[str]
    """
    information = []
    information_open = "<information>"
    information_close = "</information>"
    pos = 0
    while True:
        idx_start = s.find(information_open, pos)
        if idx_start == -1:
            break
        idx_start_content = idx_start + len(information_open)
        idx_end = s.find(information_close, idx_start_content)
        if idx_end == -1:
            break
        content = s[idx_start_content:idx_end]
        information.append(content)
        pos = idx_end + len(information_close)
    return information

def extract_smart(steps):
    outputs = []
    for step in steps:
        # step是字符串，从后往前寻找"Output"或"output"，提取它后面的内容（不包含Output或output本身）
        idx_output = step.lower().rfind("output")
        if idx_output != -1:
            idx_after = idx_output + len("output")
            outputs.append(step[idx_after:])
    return outputs

# =============== 新增：思考链路拼接（去除工具结果/特殊token） ===============

def _remove_tagged_segments(s: str, open_tag: str, close_tag: str) -> str:
    """移除字符串中所有 open_tag ... close_tag 包裹的片段（连同标签本身）。"""
    if not s:
        return s
    parts = []
    pos = 0
    while True:
        idx_start = s.find(open_tag, pos)
        if idx_start == -1:
            # 没有更多标签，追加剩余部分
            parts.append(s[pos:])
            break
        # 追加标签前的思考内容
        parts.append(s[pos:idx_start])
        idx_after_open = idx_start + len(open_tag)
        idx_end = s.find(close_tag, idx_after_open)
        if idx_end == -1:
            # 没有闭合标签，直接结束（丢弃未闭合部分）
            break
        # 跳过闭合标签后继续
        pos = idx_end + len(close_tag)
    return "".join(parts)

def _strip_special_tokens(s: str) -> str:
    """移除可能残留的特殊token。"""
    if not s:
        return s
    tokens = [
        "<result>", "</result>",
        "<information>", "</information>",
    ]
    for t in tokens:
        s = s.replace(t, "")
    return s

def extract_thinking_results(s: str) -> str:
    """
    从包含<result>...</result>片段的原始链路中，移除所有工具执行结果及其标签，
    并拼接剩余“思考”内容为一个字符串返回。
    同时清理可能残留的<result>/<information>等特殊token。
    """
    if s is None:
        return ""
    cleaned = _remove_tagged_segments(s, "<result>", "</result>")
    cleaned = _strip_special_tokens(cleaned)
    return cleaned

def extract_thinking_information(s: str) -> str:
    """
    从包含<information>...</information>片段的原始链路中，移除所有工具执行信息及其标签，
    并拼接剩余“思考”内容为一个字符串返回。
    同时清理可能残留的<result>/<information>等特殊token。
    """
    if s is None:
        return ""
    cleaned = _remove_tagged_segments(s, "<information>", "</information>")
    cleaned = _strip_special_tokens(cleaned)
    return cleaned

def extract_thinking_smart(steps):
    """
    传入steps(list[str])，每个step字符串中，从后往前寻找"Output"/"output"，
    取其之前的内容作为“思考”部分，去掉工具输出，然后将所有思考片段拼接为一个字符串返回。
    同时会移除残留的特殊token（<result>/<information>）。
    """
    if not steps:
        return ""
    thinking_parts = []
    for step in steps:
        if not isinstance(step, str):
            continue
        idx_output = step.lower().rfind("output")
        if idx_output != -1:
            thinking = step[:idx_output]
        else:
            thinking = step
        thinking_parts.append(thinking)
    combined = "".join(thinking_parts)
    combined = _strip_special_tokens(combined)
    return combined
