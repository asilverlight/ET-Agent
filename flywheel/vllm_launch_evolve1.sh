#!/bin/bash

use_qwen3=true

# Activate the Conda environment
conda activate /path/to/conda/bin/python

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs
MODEL_PATH="/path/to/qwen2.5-7b-instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"



# Launch instance 3 - using GPU 4 and 5
echo "Starting Instance 0 on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 8001 > logs/Qwen2.5-7B-Instruct_1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 0 deployed on port 8001 using GPU 0"

echo "Starting Instance 1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 8002 > logs/Qwen2.5-7B-Instruct_2.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 1 deployed on port 8002 using GPU 1"

echo "Starting Instance 2 on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 8003 > logs/Qwen2.5-7B-Instruct_2.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance 2 deployed on port 8003 using GPU 2"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID" SIGTERM
wait $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID
