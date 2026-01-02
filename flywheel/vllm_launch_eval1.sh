#!/bin/bash

# Activate the Conda environment
conda activate /path/to/conda/bin/python

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs   

# Model path - all instances use the same model
MODEL_PATH="/path/to/qwen2.5-72b-instruct"
MODEL_NAME="Qwen2.5-72B-Instruct"
# MODEL_NAME="Qwen3-4B-Instruct-2507"

# Launch Instance 1 - using GPU 0
echo "Starting Instance 1 on GPU 0"
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 4 \
    --gpu-memory-utilization 0.95 \
    --port 8005 > logs/Qwen2.5-72B-Instruct_1.log 2>&1 &
INSTANCE5_PID=$!
echo "Instance 1 deployed on port 8005 using GPU 4,5,6,7"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $INSTANCE5_PID" SIGTERM
wait $INSTANCE5_PID
