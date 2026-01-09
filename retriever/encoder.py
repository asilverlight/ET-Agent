from typing import List
import torch
import numpy as np
from utils import load_model, pooling


def parse_query(model_name, query_list, is_query=True):
    """
    processing query for different encoders
    """

    def is_zh(str):
        import unicodedata

        zh_char = 0
        for c in str:
            if "CJK" in unicodedata.name(c):
                zh_char += 1
        if zh_char / len(str) > 0.2:
            return True
        else:
            return False

    if isinstance(query_list, str):
        query_list = [query_list]

    if "e5" in model_name.lower():
        if is_query:
            query_list = [f"query: {query}" for query in query_list]
        else:
            query_list = [f"passage: {query}" for query in query_list]

    if "bge" in model_name.lower():
        if is_query:
            if is_zh(query_list[0]):
                query_list = [f"为这个句子生成表示以用于检索相关文章：{query}" for query in query_list]
            else:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

    return query_list


class Encoder:
    def __init__(self, model_name, model_path, max_length, pooling_method='mean', use_fp16=False):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode(mode=True)
    def encode(self, query_list: List[str], is_query=True, max_length=None) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, is_query)

        max_length = max_length if max_length is not None else self.max_length

        inputs = self.tokenizer(
            query_list, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb
    
    def simple_encoder(self, query: str, max_length=None):
        max_length = max_length if max_length is not None else self.max_length
        input = self.tokenizer(
            query, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input = {k: v.cuda() for k, v in input.items()}
        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((input["input_ids"].shape[0], 1), dtype=torch.long).to(
                input["input_ids"].device
            )
            output = self.model(**input, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**input, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, input["attention_mask"], self.pooling_method
            )

        query_emb = query_emb.detach()
        # .cpu().numpy()
        # query_emb = query_emb.astype(np.float32, order="C")
        return query_emb
