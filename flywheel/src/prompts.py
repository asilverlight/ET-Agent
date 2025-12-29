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

Evolve_Trajectory_Incorrect_Prompt = """## Role
You are an expert in analyzing tool-integrated reasoning trajectories.

## Task Description
You will be given a reasoning trajectory that was produced by you in a previous turn. Your goal is to examine this trajectory step by step and identify the **first** step that contains any reasoning flaw or improper tool call. Once located, provide your analysis and practical suggestions for that step.

## Flaw Categories
1. Flawed Reasoning:
   - Logical Error: The thought process is incorrect, irrelevant, or does not contribute to solving the goal.
   - Overthinking: The reasoning is unnecessarily long, repetitive, or contains redundant content.

2. Improper Tool Call:
   - Redundant Call: A tool is called even though the available information is already sufficient, or the call unnecessarily repeats prior content.
   - Inappropriate Call: The selected tool is unsuitable for the task, or the content passed to the tool is inappropriate.

## Input
### Question:
{question}

### Trajectory:
{trajectory}

## Output Requirements
Your response must contain **exactly two lines**, with no additional explanation, formatting, or text:

Line 1:
Step: [the integer index of the earliest flawed step]

Line 2:
Analysis: [a concise analysis and practical suggestions of the flaw in that step]

Among them, your analysis and practical suggestions should be written in a **single paragraph of plain text**. Do not use any headings, bullet points, numbering, or formatting. Just give a concise summary of the main flaws you found in the trajectory and how it could be improved, in continuous natural language.

For example:
Step: [an integer step number] 
Analysis: [A paragraph with analysis and practical suggestions of the flaw]"""
# 这个prompt用于找到第一个出错的步骤，然后给出其反思内容

Evolve_Trajectory_Incorrect_Generate_New_Step_Prompt = """## Role:
You are an expert in refining and completing tool-integrated reasoning trajectories.

## Task Description:
You will be given:
1. A question
2. An incomplete trajectory
3. An analysis specifically about the **last step** in the trajectory

Your goal is to generate a **new, corrected final step** to replace the current last step based on the provided analysis. Only generate a new final step, do not generate any other content.

## Input Data:
### Question: {question}
### Trajectory (incomplete):
{trajectory}
### Analysis of the last step:
{analysis}

## Output Format:
Your output must be exactly one line containing only the modified final step, in full.

For example:
your reasoning process. <search> search query here </search> <result> search result here </result> your reasoning process. <python> python code here </python> <result> python interpreter result here </result> your reasoning process. <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>.

Your modified step: """

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
# 这个prompt用于反思精炼正确的trajectory中，第一处不合理的step

Search_Hints = [
    " Wait a minute, I might need to rethink this problem and then initiate a new query.",
    " Hold on, I think I should reconsider this question and generate a new query.",
    " But wait, I think I need to rethink how to solve it and then try a new search.",
    " Wait, maybe I should rethink this question and conduct a new search.",
    " But wait, maybe I should reconsider my approach and run a new search.",
    " Wait, I think rethinking the problem first and launching another query would help.",
]
Python_Hints = [
    " Wait, perhaps I should use Python tools and rethink this problem.",
    " Wait, I think I should turn to Python tools and reassess the problem.",
    " Wait a moment, I might need to use Python tools and rethink this problem.",
    " But wait, I'll probably need a Python tool to help me reconsider this question.",
    " Hold on, I may need to use a Python tool to reconsider this question.",
    " Wait, I think I should revisit this problem and use a Python tool to help."
]


Analyze_Incorrect_Trajectory_Prompt = """## Role
You are an expert in analyzing tool-integrated reasoning trajectories.

## Task Description
You will receive a reasoning trajectory that was produced by you in a previous turn. Your goal is to review your trajectory step by step, identify any flaws in reasoning or tool usage, and provide a clear analysis and practical improvements of the flaws you find.

The flaw categories include flawed reasoning (The reasoning is incorrect, irrelevant, or does not contribute to solving the problem) and improper tool call (The tool is unsuitable for the task, or the input passed to the tool is inappropriate).

## Output Requirements
Write your response as a **single paragraph of plain text**. Do not use any headings, bullet points, numbering, or formatting. Just give a concise summary of the main flaws you found in the trajectory and how it could be improved, in continuous natural language.

## Input
### Question:
{question}

### Trajectory:
{trajectory}

Your output:"""
# 这个prompt用于对整条轨迹做分析修正

Rewrite_Trajectory_Whole_Level_Prompt = """You are an expert in trajectory refinement. You will be given a **question** and a corresponding **trajectory** that uses a fixed format with tags such as `<think>`, `<search>`, `<result>`, `<python>`, and `<answer>`.

Your task is to **refine ONLY the text inside `<think>...</think>` tags**, while keeping the trajectory's structure, tag order, and all other content unchanged.

### Requirements:

1. **Do NOT add, remove, reorder, or rename any tags.**
   The sequence of tags must remain exactly the same as in the original trajectory.
   For example, if the format is:
   `<think>...</think> <search>...</search> <result>...</result> <think>...</think> <answer>...</answer>`
   then the refined version must follow the exact same pattern.

2. **Only modify the content inside `<think>` tags.**

   * Remove redundant thoughts or irrelevant reasoning.
   * Eliminate obvious mistakes or unnecessary repetition.
   * Preserve the core reasoning and logical flow.
   * Keep the refined version concise, clear, and logically consistent.

3. **Do NOT alter anything inside `<search>`, `<result>`, `<python>`, or `<answer>` tags.**
   Their content, formatting, and text must remain identical.

4. **Keep the formatting and tags unchanged.**

---

### Input

**[Question]**
{question}

**[Original Trajectory]**
{trajectory}

---

### Output

Provide the **refined trajectory**, keeping the same structure and identical content outside the `<think>` tags, with only streamlined and corrected content inside each `<think>` section.

Your output:"""
# 这个prompt用于对整条轨迹做精炼重写