import sys
import os
sys.path.append(os.getcwd())

import re
import string
from typing import Dict, Any, List, Union, Optional, Tuple, Set
from collections import Counter

from .math_equivalence import is_equiv


def normalize_answer(text: str, remove_articles: bool = False, remove_punctuations: bool = False) -> str:
    """
    Normalize answer text.

    Args:
        text: The original answer text.
        remove_articles: Whether to remove articles (a/an/the).
        remove_punctuations: Whether to remove punctuation marks.

    Returns:
        The normalized answer.
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower().strip()
    
    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    
    # Remove punctuation
    if remove_punctuations:
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
    
    # Fix extra spaces
    return " ".join(text.split())


def compute_token_overlap(prediction: str, reference: str) -> Tuple[int, int, int]:
    """
    Compute token overlap between prediction and reference.

    Args:
        prediction: Predicted answer.
        reference: Reference answer.

    Returns:
        A tuple of (number of overlapping tokens, number of prediction tokens, number of reference tokens).
    """
    prediction_tokens = prediction.split()
    reference_tokens = reference.split()
    
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    
    return num_same, len(prediction_tokens), len(reference_tokens)


def compute_f1_score(num_same: int, pred_len: int, ref_len: int) -> float:
    """
    Compute F1 score.

    Args:
        num_same: Number of overlapping tokens.
        pred_len: Number of prediction tokens.
        ref_len: Number of reference tokens.

    Returns:
        F1 score (0-1).
    """
    if num_same == 0:
        return 0.0

    precision = num_same / pred_len if pred_len > 0 else 0
    recall = num_same / ref_len if ref_len > 0 else 0

    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_math_prediction(
    prediction: str,
    reference: str
) -> Dict[str, Union[int, float]]:
    """
    Evaluate mathematical prediction result.

    Args:
        prediction: Predicted answer.
        reference: Reference answer.

    Returns:
        A dictionary containing evaluation metrics.
    """
    # Normalize answers
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)

    # Compute exact match and accuracy
    em = int(normalized_prediction == normalized_reference)
    acc = int(normalized_reference in normalized_prediction)
    
    # Compute F1 score
    num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
    f1 = compute_f1_score(num_same, pred_len, ref_len)

    math_equal = int(is_equiv(normalized_prediction, normalized_reference))

    return {
        "em": em,
        "acc": acc,
        "f1": f1,
        "math_equal": math_equal
    }


def evaluate_qa_prediction(
    prediction: str,
    references: Union[str, List[str]]
) -> Dict[str, Union[int, float]]:
    """
    Evaluate QA prediction result.

    Args:
        prediction: Predicted answer.
        references: A single reference or a list of reference answers.

    Returns:
        A dictionary containing evaluation metrics.
    """
    # Ensure references is a list
    if isinstance(references, str):
        if not references.startswith("["):
            references = [references]
        else:
            # Try parsing string list
            references = [e.strip() for e in re.split(r",\s*", references.strip('[]'))]

    # Initialize result
    result = {"em": 0, "acc": 0, "f1": 0, "math_equal": 0}
    
    # Normalize prediction (stricter normalization for QA)
    normalized_prediction = normalize_answer(prediction, remove_articles=True)

    # print('==============')
    # print(references)
    # print('==============')

    # Compute metrics for each reference and take the highest
    # try:
    # 如果references里的元素都是bool类型，则将其都转化为str
    if isinstance(references, list) and all(isinstance(r, bool) for r in references):
        references = [str(r) for r in references]
    elif isinstance(references, bool):
        references = [str(references)]
    try:
        for reference in references:
            normalized_reference = normalize_answer(reference, remove_articles=True)

            em = int(normalized_prediction == normalized_reference)
            acc = int(normalized_reference in normalized_prediction)
            
            num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
            f1 = compute_f1_score(num_same, pred_len, ref_len)

            result["em"] = max(result["em"], em)
            result["acc"] = max(result["acc"], acc)
            result["f1"] = max(result["f1"], f1)

            math_equal = int(is_equiv(normalized_prediction, normalized_reference))
            result["math_equal"] = max(result["math_equal"], math_equal)
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Prediction: {prediction}")
        print(f"References: {references}")
        assert False

    return result
