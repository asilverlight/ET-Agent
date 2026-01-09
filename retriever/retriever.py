import json
import os
import warnings
from typing import List, Dict
import functools
from tqdm import tqdm
import faiss

from utils import load_corpus, load_docs
from encoder import Encoder

def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num=None):
        if num is None:
            num = self.topk
        if self.use_cache:
            if isinstance(query_list, str):
                new_query_list = [query_list]
            else:
                new_query_list = query_list

            no_cache_query = []
            cache_results = []
            for query in new_query_list:
                if query in self.cache:
                    cache_res = self.cache[query]
                    if len(cache_res) < num:
                        warnings.warn(
                            f"The number of cached retrieval results is less than topk ({num})"
                        )
                    cache_res = cache_res[:num]
                    # separate the doc score
                    doc_scores = [item.pop("score") for item in cache_res]
                    cache_results.append((cache_res, doc_scores))
                else:
                    cache_results.append(None)
                    no_cache_query.append(query)

            if no_cache_query != []:
                # use batch search without decorator
                no_cache_results, no_cache_scores = (
                    self._batch_search_with_rerank(no_cache_query, num, True)
                )
                no_cache_idx = 0
                for idx, res in enumerate(cache_results):
                    if res is None:
                        assert (
                            new_query_list[idx] == no_cache_query[no_cache_idx]
                        )
                        cache_results = (
                            no_cache_results[no_cache_idx],
                            no_cache_scores[no_cache_scores],
                        )
                        no_cache_idx += 1

            results, scores = (
                [t[0] for t in cache_results],
                [t[1] for t in cache_results],
            )

        else:
            results, scores = func(self, query_list, num, True)

        if self.save_cache:
            # merge result and score
            save_results = results.copy()
            save_scores = scores.copy()
            if isinstance(query_list, str):
                query_list = [query_list]
                if "batch" not in func.__name__:
                    save_results = [save_results]
                    save_scores = [save_scores]
            for query, doc_items, doc_scores in zip(
                query_list, save_results, save_scores
            ):
                for item, score in zip(doc_items, doc_scores):
                    item["score"] = score
                self.cache[query] = doc_items

        return results

    return wrapper


def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num=None):
        results, scores = func(self, query_list, num)
        return results

    return wrapper


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config["retrieval_method"]
        self.topk = config["retrieval_topk"]

        self.index_path = config["index_path"]
        self.corpus_path = config["corpus_path"]

        self.save_cache = config["save_retrieval_cache"]
        self.use_cache = config["use_retrieval_cache"]
        self.cache_path = config["retrieval_cache_path"]


        # if self.save_cache:
        #     self.cache_save_path = os.path.join(
        #         config["save_dir"], "retrieval_cache.json"
        #     )
        #     self.cache = {}
        # if self.use_cache:
        #     assert self.cache_path is not None
        #     with open(self.cache_path, "r") as f:
        #         self.cache = json.load(f)


    def _search(
        self, query: str, num: int, return_score: bool=False
    ) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    def _batch_search(self, query_list, num):
        pass

    # @cache_manager
    # @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    # @cache_manager
    # @rerank_manager
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    # @rerank_manager
    # def _batch_search_with_rerank(self, *args, **kwargs):
    #     return self._batch_search(*args, **kwargs)

    # @rerank_manager
    # def _search_with_rerank(self, *args, **kwargs):
    #     return self._search(*args, **kwargs)



class DenseRetriever(BaseRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        # print(self.index_path)
        self.index = faiss.read_index(self.index_path)
        
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True

        # res = faiss.StandardGpuResources()
        # res.setTempMemory(256 * 1024 * 1024)
        # gpu_id = 0

        self.corpus = load_corpus(self.corpus_path, config['retrieval_method'])
        self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        # self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)

        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config["retrieval_model_path"],
            pooling_method=config["retrieval_pooling_method"],
            max_length=config["retrieval_query_max_length"],
            use_fp16=config["retrieval_use_fp16"],
        )
        self.topk = config["retrieval_topk"]
        self.batch_size = self.config["retrieval_batch_size"]

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=30)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        
        # 筛选文档：优先选择句子数量大于等于3的文档
        valid_indices = []
        short_indices = []
        
        for i, result in enumerate(results):
            if 'text' in result:
                sentences = result['text']
                if len(sentences) >= 400:
                    valid_indices.append(i)
                else:
                    short_indices.append(i)
            else:
                content_str = result['contents']
                split_index = content_str.find('\n')
                content = content_str[split_index+1:]
                content_sentences = content
                if len(content_sentences) >= 400:
                    valid_indices.append(i)
                else:
                    short_indices.append(i)
        
        # 如果有效文档不足num个，则从短文档中补充
        selected_indices = valid_indices[:num]
        if len(selected_indices) < num:
            selected_indices.extend(short_indices[:num-len(selected_indices)])
        
        # 确保只取num个文档
        selected_indices = selected_indices[:num]
        
        # 准备输出
        outputs = []
        selected_scores = []
        
        for i in selected_indices:
            result = results[i]
            if 'text' in result:
                sentences = result['text']
                sentences = sentences[:750]
                content = sentences + '...'
                title = result['title']
            else:
                content_str = result['contents']
                split_index = content_str.find('\n')
                title = content_str[:split_index]
                content = content_str[split_index+1:]
                content_sentences = content
                content_sentences = content_sentences[:750]
                content = content_sentences + '...'
            
            outputs.append(f'{title}\n{content}\n\n')
            selected_scores.append(scores[i])
        
        if return_score:
            return outputs, selected_scores
        return outputs

    def _batch_search(
        self, query_list: List[str], num: int = None
    ):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        batch_size = self.batch_size

        results = []
        scores = []

        for start_idx in tqdm(
            range(0, len(query_list), batch_size), desc="Retrieval process: "
        ):
            query_batch = query_list[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            # print(batch_idxs)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            # print(batch_scores)
            # print(batch_idxs)
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [
                batch_results[i * num : (i + 1) * num]
                for i in range(len(batch_idxs))
            ]

            scores.extend(batch_scores)
            results.extend(batch_results)

        outputs = []#返回一个list，每个是一个string
        for i in range(len(results)):
            output = ''
            contents = []
            titles = []
            for result in results[i]:
                sentences = result['text']
                sentences = sentences[:1000]
                sentences = '. '.join(sentences)    
                contents.append(sentences)
                titles.append(result['title'])  
            for j in range(len(contents)):
                output += f"Doc {j+1}: {titles[j]}\n{contents[j]}\n\n"
            outputs.append(output)
        return outputs
