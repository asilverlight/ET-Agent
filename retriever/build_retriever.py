from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union, Optional
import asyncio
from collections import deque

# from flashrag.config import Config
# from utils import get_retriever
import sys
import sys
from utils import get_retriever
app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None

def init_retriever(args):
    global retriever_semaphore
    retriever_config = {
        'retrieval_topk': 4,
        'retrieval_method': 'e5',
        'retrieval_model_path': "/path/to/E5-base-v2",
        'index_path': '/path/to/e5_Flat.index',
        'corpus_path': '/path/to/wiki_dump.jsonl',
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'retrieval_pooling_method': 'mean',
        'retrieval_query_max_length': 128,
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 256,
    }
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(retriever_config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 4
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 4

# 定义响应模型
class SearchResponse(BaseModel):
    results: List[str]
    scores: Optional[List[float]] = None  # scores 是可选的，仅当 return_score=True 时返回

@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].search(query, top_n, return_score)
                return {"results": results, "scores": scores}  # 返回两个列表
            results = retriever_list[retriever_idx].search(query, top_n)
            return {"results": results}  # 只返回一个列表
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=List[str])
async def batch_search(request: BatchQueryRequest):
    print(f'received batch search request: {request}')
    query = request.query
    top_n = request.top_n

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].batch_search(query, top_n)
            return results
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_retriever", 
        type=int, 
        default=1,
        help="number of retriever to use, more retriever means more memory usage and faster retrieval speed"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=1557,
        help="port to use for the serving"
    )
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
