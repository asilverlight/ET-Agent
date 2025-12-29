#!/bin/bash

use_qwen3=true

# Activate the Conda environment
source /path/to/conda/bin/activate
conda activate arpo

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs

MODEL_PATH="/path/to/your_model_path"
MODEL_NAME="your_model_name"


# Launch instance 3 - using GPU 4 and 5
echo "Starting Instance 0 on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8011 \
    --no-enable-prefix-caching > logs/model_1.log 2>&1 &
INSTANCE0_PID=$!
echo "Instance 0 deployed on port 8011 using GPU 0"

echo "Starting Instance 1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8012 \
    --no-enable-prefix-caching > logs/model_1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 1 deployed on port 8012 using GPU 1"

echo "Starting Instance 2 on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8013 \
    --no-enable-prefix-caching > logs/model_1.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 2 deployed on port 8013 using GPU 2"

echo "Starting Instance 3 on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8014 \
    --no-enable-prefix-caching > logs/model_1.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance 4 deployed on port 8014 using GPU 3"
# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE0_PID $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID" SIGTERM
wait $INSTANCE0_PID $INSTANCE1_PID $INSTANCE2_PID $INSTANCE3_PID
