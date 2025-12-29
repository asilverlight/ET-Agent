import sys
import os

sys.path.append(os.getcwd())

import json
from typing import List, Tuple

from .utils import extract_solution, last_boxed_only_string, remove_boxed


class DataLoader:
    """Loading Datasets"""

    def __init__(self, args):
        """
        Args:
            dataset_name
            data_path
        """
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.data_path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')
        self.counts = args.counts

    def load_data(self) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
        """
        Load dataset

        Returns:
            (questions, answers, choices, formats)
        """
        questions = []
        answers = []
        choices = []
        formats = []

        print(f"Loading dataset from {self.data_path}")

        if (
            "aime" in self.dataset_name
            or self.dataset_name == "amc23"
            or self.dataset_name == "gsm8k"
            or "tabmwp" == self.dataset_name
            or "gaokao2023en" == self.dataset_name
            or "college_math" == self.dataset_name
        ):
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data.get("question", data.get("problem")))
                    answer = data["answer"]
                    if "gsm8k" in self.data_path:
                        answer = extract_solution(answer)
                    answers.append(answer)
        elif "svamp" == self.dataset_name or "asdiv" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    body = data["body"] if "body" in data else data["Body"]
                    question = (
                        data["question"] if "question" in data else data["Question"]
                    )
                    answer = data["answer"] if "answer" in data else data["Answer"]
                    if "asdiv" in self.data_path:
                        answer = answer.split(" (")[0]
                    questions.append(body + " " + question)
                    answers.append(answer)
        elif "mawps" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["input"])
                    answers.append(data["target"])
        elif "carp_en" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["content"])
                    answers.append(data["answer"])
        elif "minerva_math" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    question = data["problem"]
                    answer = data["solution"]
                    try:
                        answer = remove_boxed(last_boxed_only_string(answer))
                    except:
                        pass
                    questions.append(question)
                    answers.append(answer)
        elif "olympiadbench" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["final_answer"][0])
        elif "math" == self.dataset_name or "aime25" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["problem"])
                    answers.append(data["answer"])
        elif "gaia" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["Question"])
                    answers.append(data["answer"])
        elif "csbench" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["Question"])
                    answers.append(data["Answer"])
                    formats.append(data['Format'])
                    if data['Format'] == "Multiple-choice":
                        choices.append(
                            [data['A'], data['B'], data['C'], data['D']]
                        )
                    else:
                        choices.append([])
        elif "mmlu-pro" == self.dataset_name:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["answer"])
                    choices.append(data['options'])
                    formats.append("Multiple-choice")
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["answer"])
        if len(choices) == 0:
            choices = [[]] * len(questions)
        if len(formats) == 0:
            formats = [None] * len(questions)

        print(f"Loading {len(questions)} samples from {self.data_path}...")
        questions = questions[:self.counts]
        answers = answers[:self.counts]
        choices = choices[:self.counts]
        formats = formats[:self.counts]
        return questions, answers, choices, formats
