#!/bin/bash

# Activate the Conda environment
source /path/to/conda/bin/activate
conda activate arpo

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs

MODEL_PATH="/path/to/qwen2.5-72b-instruct"
MODEL_NAME="Qwen2.5-72B-Instruct"

# Launch Instance 1 - using GPU 0
echo "Starting Instance 1 on GPU 0,1,2,3"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 4 \
    --gpu-memory-utilization 0.95 \
    --port 8031 > logs/reasoning_model1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 1 deployed on port 8031 using GPU 0,1,2,3"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $INSTANCE1_PID" SIGTERM
wait $INSTANCE1_PID
