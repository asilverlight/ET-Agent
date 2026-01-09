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

def process_evolve_trajectory_input_correct(tokenizer, input, trajectory):
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

def process_evolve_trajectory_input_incorrect(tokenizer, input, trajectory):
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

def process_evolve_trajectory_input_hint_level(tokenizer, input, trajectory, source):
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