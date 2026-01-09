#!/bin/bash
# conda activate vllm_arpo


export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /path/to/qwen2.5-72b-instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 4 \
  --gpu-memory-utilization 0.95 \
  --port 8051 \
  --max-logprobs 100