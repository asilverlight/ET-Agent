import json
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str, retrieval_method: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", cache_dir=f'/path/to/cache/{retrieval_method}/')
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results

import re
import importlib
import torch
import torch.nn.functional as F
from math import exp
def get_dataset(config, data_dir='data_dir', value='question'):
    
    data_path = config[data_dir]
    questions = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            questions.append(data[value])
    return questions

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")
    
def remove_substring(s, substring):
    # 使用正则表达式匹配子串及其左右可能存在的空行
    pattern = re.compile(r'\n*\s*' + re.escape(substring) + r'\s*\n*', re.IGNORECASE)
    # 替换匹配到的子串及其左右空行为单个空行
    modified_s = re.sub(pattern, '\n', s)
    return modified_s.strip()  # 去除字符串首尾的空行
    
def retain_after_last_substring(s, substring):
    # 查找最后一个"user\n"子串的位置
    index = s.rfind(substring)
    
    # 如果找到了该子串，则截取该位置之后的所有字符
    if index != -1 and index + len(substring) < len(s):
        return s[index + len(substring):]
    else:
        # 如果没有找到该子串，则返回原字符串
        return s
    
def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("retriever"), "BM25Retriever")(config)
    else:
        return getattr(importlib.import_module("retriever"), "DenseRetriever")(config)