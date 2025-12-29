from .prompts import *
import random
from .utils import *

def process_generate_initial_trajectory_input(tokenizer, input):
    generate_initial_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "system", "content": ToolStar_Prompt
            },
            {
                "role": "user", "content": input
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return generate_initial_trajectory_context

def process_judge_correctness_input(tokenizer, input, labeled_answer, pred_answer):
    judge_correctness_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Judge_Correctness_Prompt.format(question=input, labeled_answer=labeled_answer, pred_answer=pred_answer)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return judge_correctness_context

def process_modify_previous_trajectory_input(tokenizer, input, trajectory):
    modify_previous_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "system", "content": ToolStar_Prompt
            },
            {
                "role": "user", "content": input
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    modify_previous_trajectory_context += f"\n\n{trajectory}"
    return modify_previous_trajectory_context

def process_input_correct_step_refine(tokenizer, input, trajectory):
    # 当正确时，修正某个步骤
    trajectory = split_trajectory_to_steps(trajectory)
    evolve_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Evolve_Trajectory_Correct_Prompt.format(question=input, trajectory=trajectory)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return evolve_trajectory_context

def process_input_correct_rewrite_whole_trajectory(tokenizer, input, trajectory):
    # 当正确时，重写整条轨迹
    rewrite_whole_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Rewrite_Trajectory_Whole_Level_Prompt.format(question=input, trajectory=trajectory)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return rewrite_whole_trajectory_context

def process_input_incorrect_step_refine(tokenizer, input, trajectory):
    # 当错误时，修正某个步骤
    trajectory = split_trajectory_to_steps(trajectory)
    evolve_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Evolve_Trajectory_Incorrect_Prompt.format(question=input, trajectory=trajectory)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return evolve_trajectory_context

def process_input_incorrect_generate_new_step(tokenizer, input, trajectory, analysis_content):
    # 根据analyze，生成一个新步骤
    trajectory = split_trajectory_to_steps(trajectory)
    evolve_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Evolve_Trajectory_Incorrect_Generate_New_Step_Prompt.format(question=input, trajectory=trajectory, analysis=analysis_content)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return evolve_trajectory_context

def process_input_incorrect_insert_hint_tail(tokenizer, input, trajectory, source):
    # 当错误时，在步骤的末尾插入hint修正
    evolve_hint_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "system", "content": ToolStar_Prompt
            },
            {
                "role": "user", "content": input
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    if '<answer>' in trajectory:
        # 找到第一个'<answer>'的位置，并连同'<answer>'去掉
        answer_idx = trajectory.find('<answer>')
        trajectory = trajectory[:answer_idx].strip()
        if trajectory.endswith("</think>"):
            trajectory = trajectory[:-len("</think>")]
        else:
            trajectory += "<think>"
    if source in ['openr1', 'numina-tir', 'numina-cot', 'aime', 'dapo']:
        trajectory += random.choice(Python_Hints)
    else:
        trajectory += random.choice(Search_Hints)
    evolve_hint_trajectory_context += f"\n\n{trajectory}"
    return evolve_hint_trajectory_context, trajectory

def process_input_incorrect_insert_analysis_tail(tokenizer, input, trajectory):
    # 当整条轨迹错误时，对整条轨迹进行分析，然后再末尾插入分析
    refine_whole_trajectory_context = tokenizer.apply_chat_template(
        [
            {
                "role": "user", "content": Analyze_Incorrect_Trajectory_Prompt.format(question=input, trajectory=trajectory)
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    refine_whole_trajectory_context += f"\n\nLet me check my trajectory."
    return refine_whole_trajectory_context