#!/bin/bash

# Change to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Set Python environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Evaluation configuration parameters
USE_LLM=$USE_LLM                               # Whether to use LLM for equivalence evaluation
API_BASE_URL="http://localhost:8051/v1"   # Base URL of the LLM API
MODEL_NAME="Qwen2.5-72B-Instruct"         # Name of the LLM evaluation model
CONCURRENT_LIMIT=50                        # Limit of concurrent evaluations
TIMEOUT=36                              # Total evaluation timeout (seconds)

# Build command line arguments
CMD="python evaluate.py"
CMD+=" --output_path $OUTPUT_PATH"
CMD+=" --task $TASK"
CMD+=" --concurrent_limit $CONCURRENT_LIMIT"
CMD+=" --timeout $TIMEOUT"
CMD+=" --counts $COUNTS"

# If LLM evaluation is enabled, add related parameters
if [ "$USE_LLM" = true ]; then
    CMD+=" --use_llm"
    
    # Add API_BASE_URL parameter if not empty
    if [ ! -z "$API_BASE_URL" ]; then
        CMD+=" --api_base_url $API_BASE_URL"
    fi
    
    # Add MODEL_NAME parameter if not empty
    if [ ! -z "$MODEL_NAME" ]; then
        CMD+=" --model_name $MODEL_NAME"
    fi
fi

# Execute command
echo "Executing evaluation command: $CMD"
eval $CMD | tee logs/evaluate.log 2>&1