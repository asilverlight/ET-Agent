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
MODEL_PATH="/path/to/your_model_path"
MODEL_NAME="YOUR_MODEL_NAME"



# Launch instance 3 - using GPU 4 and 5
echo "Starting Instance 0 on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 9001 > logs/YOUR_MODEL_NAME_1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 0 deployed on port 9001 using GPU 0"

echo "Starting Instance 1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 9002 > logs/YOUR_MODEL_NAME_2.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 1 deployed on port 9002 using GPU 1"

echo "Starting Instance 2 on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 9003 > logs/YOUR_MODEL_NAME_3.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance 2 deployed on port 9003 using GPU 2"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID" SIGTERM
wait $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID
