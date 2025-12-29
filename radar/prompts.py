Prompt_redundancy = """You are an expert analyst of tool-integrated reasoning tasks. 
Given a question, its golden answer, and a reasoning trajectory, your task is to examine the following content and determine whether the trajectory contains any redundant or ineffective tool calls.

[Question]
{question}

[Golden Answer]
{answer}

[Trajectory]
{trajectory}

Definitions:
- Redundant tool call: A call whose result is repetitive of previous content, unused in later reasoning, or does not contribute to the final answer.
- Ineffective tool call: A call that returns an irrelevant or unhelpful result providing no meaningful contribution to the reasoning process.
Your tasks:
1. Determine whether any redundant or ineffective tool calls exist in the trajectory.
2. If such calls exist, output only "<answer>yes</answer>".
3. If none exist, output only "<answer>no</answer>".
Output format:
Only output "yes" or "no", enclosed within <answer> and </answer> tags, and do not output anything else."""