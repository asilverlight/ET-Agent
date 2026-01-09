# from efficiency_rl.make_sft_datas3.src.prompts import Evolve_Strategy_Prompt


ToolStar_Prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format."""

Judge_Correctness_Prompt = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with "Correct" if the prediction is correct and "Incorrect" otherwise. Do not output anything else.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {labeled_answer}
Predicted Answer: {pred_answer}"""

Evolve_Trajectory_Incorrect_Prompt = """## Role:
You are an expert in analyzing tool-integrated reasoning trajectory. 

## Task Description: 
Your goal is to analyze a given trajectory to identify and locate the first occurrence with any reasoning flaws or improper tool usage. You should find the earliest step that causes the flaw, and provide an analysis and a corrected step for that step.

## Flaw Definitions:
1. Flawed Reasoning:
    - Logical Error: The thought process is incorrect or irrelevant to solving the final goal.
    - Overthinking: The thinking process is excessively lengthy and complex, containing unnecessary steps and even instances of repeating the same content.
2. Improper Tool Call:
    - Redundant Call: A tool is called when the necessary information is already sufficient, or the tool call's content is repeated.
    - Inappropriate Call: The wrong tool is selected for the task, or the tool call's content is inappropriate.

## Input Data:
### Question: {question}
### Trajectory: 
{trajectory}

## Output Format:
Your output must be exactly three lines. The first line is the earliest step number with the flaw, the second line is the analysis of the flaw in that step, and third second line is the corrected step. In this scenario, the output format is as follows:
Step: [an integer step number]
Analysis: [A concise analysis of the flaw in that step]
Corrected Step: [the full, corrected step string]

For example:

Step: [an integer step number]
Analysis: [A concise analysis of the flaw in that step]
Corrected Step: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>."""

Evolve_Trajectory_Correct_Prompt = """## Role:
You are an expert in analyzing tool-integrated reasoning trajectory. 

## Task Description:
Your goal is to analyze a given trajectory to identify and locate the first occurrence with any redundant reasoning or redundant tool usage. If any redundancy is identified, you should find the earliest step that exists redundancy, and provide an analysis and a simplified, more efficient version for that step.

## Redundancy Definitions:
1. Redundant Reasoning: The thought process includes unnecessary, overly complex, verbose, or repetitive content.
2. Redundant Tool Call: A tool is called when the necessary information is already sufficient, or the tool call's content is highly repetitive of a prior one seeking no new information.
  
## Input Data:
### Question: {question}
### Trajectory: 
{trajectory}

## Output Format:
Your output must be one of the following two formats, with no additional explanations or text.
**Scenario 1: The trajectory does not have any redundant reasoning or redundant tool usage.**
Only output 'Step: no', and do not output anything else.
**Scenario 2: The trajectory exists any redundant reasoning or redundant tool usage.**
Your output must be exactly three lines. The first line is the earliest step number with redundancy, the second line is the analysis of the redundancy in that step, and the third line is a simplified, more efficient version of that specific step. In this scenario, the output format is as follows:
Step: [an integer step number]
Analysis: [A concise analysis of the redundancy in that step]
Corrected Step: [the full, simplified step string]

For example:
In scenario 1, the output is 'Step: no'.

In scenario 2, the output format is as follows:
Step: [an integer step number]
Analysis: [A concise analysis of the redundancy in that step]
Corrected Step: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>."""

Search_Hints = [
    " Wait a minute, I might need to rethink this problem and then initiate a new query.",
    " Hold on, I think I should reconsider this question and generate a new query.",
    " But wait, I think I need to rethink how to solve it and then try a new search.",
    " Wait, maybe I should rethink this question and conduct a new search."
]
Python_Hints = [
    " Wait, perhaps I should use Python tools and rethink this problem.",
    " Wait a moment, I might need to use Python tools and rethink this problem.",
    " But wait, I'll probably need a Python tool to help me reconsider this question.",
    " Hold on, I may need to use a Python tool to reconsider this question."
]