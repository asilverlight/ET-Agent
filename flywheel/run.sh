#!/bin/bash

# Activate the Conda environment
# conda activate evaluation
# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Set Python environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Reasoning model endpoints
m1_endpoints=(
    "http://localhost:8001/v1"
    "http://localhost:8002/v1"
    "http://localhost:8003/v1"
)  
m2_endpoints=(
    "http://localhost:8001/v1"
    "http://localhost:8002/v1"
    "http://localhost:8003/v1"
)
m3_endpoints=(
    "http://localhost:8005/v1"
)
# 这是训好的推理模型
ENDPOINTS_M1=$(echo "${m1_endpoints[@]}" | tr '\n' ' ')
ENDPOINTS_M2=$(echo "${m2_endpoints[@]}" | tr '\n' ' ')
ENDPOINTS_M3=$(echo "${m3_endpoints[@]}" | tr '\n' ' ')

SAMPLE_TIMEOUT=1200  # Timeout for one sample

EXP_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH_M1="/path/to/qwen2.5-7b-instruct"
MODEL_PATH_M2="/path/to/qwen2.5-7b-instruct"
MODEL_PATH_M3="/path/to/qwen2.5-72b-instruct"

with_tools=true
if [ "$with_tools" = true ]; then
    PROMPT_TYPE="TIR"          # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=6                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=6                # Max search tool invocation times
    ADDITIONAL_PYTHON_TIMES=4
    ADDITIONAL_SEARCH_TIMES=4
else
    PROMPT_TYPE="base"                 # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=0                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=0                 # Max search tool invocation times
fi


# VLLM config
echo "Inference endpoints: $ENDPOINTS_INFER"
echo "Evaluation endpoints: $ENDPOINTS_EVAL"
API_KEYS=""                     # API keys list, corresponds to endpoints; empty means default "EMPTY"
DEFAULT_MODEL_M1="Qwen2.5-7B-Instruct"  # Default model name
DEFAULT_MODEL_M3="Qwen2.5-72B-Instruct"  # Default model name
DEFAULT_MODEL_M2="Qwen2.5-7B-Instruct"  # Default model name

# Generation parameters
TEMPERATURE=1                     # Temperature parameter
MAX_TOKENS=4096                     # Max tokens to generate
TOP_P=0.95                          # Top-p truncation
TOP_K=20                           # Top-k truncation
MIN_P=0.0                          # Minimum probability threshold
REPETITION_PENALTY=1.1             # Repetition penalty factor
INCLUDE_STOP_STR=true              # Whether to include stop string in output

# Inference configuration
BATCH_SIZE=8                       # Batch size
MAX_CONCURRENT=5                  # Max concurrent requests
COUNTS=20                        # Number of samples to process

# Tool configurations
CONDA_ENV="/path/to/conda/bin/python"                                # Conda environment name
PYTHON_MAX_CONCURRENT=32                        # Max concurrent Python executor
BING_API_KEY="your_bing_api_key"  # Bing Search API key
BING_ZONE="your_bing_zone"                        # Bing search zone
SEARCH_MAX_RESULTS=5                            # Max number of search results
SEARCH_RESULT_LENGTH=1000                        # Max length per search result
BING_REQUESTS_PER_SECOND=30.0                    # Max Bing search requests per second
BING_MAX_RETRIES=3                              # Max Bing search retries
BING_RETRY_DELAY=1.0                            # Bing search retry delay (seconds)

SUMM_MODEL_URLS=$ENDPOINTS_M3
SUMM_MODEL_NAME=$DEFAULT_MODEL_M3
SUMM_MODEL_PATH=$MODEL_PATH_M3
# 这是总结摘要模型
SEARCH_CACHE_FILE="/path/to/search_cache.db"
URL_CACHE_FILE="/path/to/search_url_cache.db"
MAX_EVOLVE_TRAJECTORIES_TIMES=3
# USE_LOG=true
USE_LOG=false

# USE_LOCAL_SEARCH=false
USE_LOCAL_SEARCH=true
LOCAL_SEARCH_URL="your_local_search_url"
COMPATIBLE_SEARCH=false # 兼容搜索代表根据问题种类选择local search或者websearch
echo "Processing dataset: $DATA_PATH"
echo "USE_LOCAL_SEARCH: $USE_LOCAL_SEARCH, USE_SDS: $USE_SDS"

# Build command line arguments
DATA_PATH="/path/to/data.json"
OUTPUT_PATH="/path/to/output.json"
CMD="python -u run.py"
CMD+=" --endpoints_m1 $ENDPOINTS_M1"
CMD+=" --endpoints_m2 $ENDPOINTS_M2"
CMD+=" --endpoints_m3 $ENDPOINTS_M3"
CMD+=" --model_path_m1 $MODEL_PATH_M1"
CMD+=" --model_path_m2 $MODEL_PATH_M2"
CMD+=" --model_path_m3 $MODEL_PATH_M3"
CMD+=" --default_model_m1 $DEFAULT_MODEL_M1"
CMD+=" --default_model_m2 $DEFAULT_MODEL_M2"
CMD+=" --default_model_m3 $DEFAULT_MODEL_M3"
# If API_KEYS is not empty, add the parameter
if [ ! -z "$API_KEYS" ]; then
    CMD+=" --api_keys $API_KEYS"
fi

# Add generation parameters
CMD+=" --temperature $TEMPERATURE"
CMD+=" --max_tokens $MAX_TOKENS"
CMD+=" --top_p $TOP_P"
CMD+=" --top_k $TOP_K"
CMD+=" --min_p $MIN_P"
CMD+=" --repetition_penalty $REPETITION_PENALTY"
CMD+=" --include_stop_str_in_output $INCLUDE_STOP_STR"

# Add inference config parameters
CMD+=" --max_concurrent_requests $MAX_CONCURRENT"
CMD+=" --output_path $OUTPUT_PATH"
CMD+=" --prompt_type $PROMPT_TYPE"
CMD+=" --counts $COUNTS"
CMD+=" --max_python_times $MAX_PYTHON_TIMES"
CMD+=" --max_search_times $MAX_SEARCH_TIMES"
CMD+=" --additional_python_times $ADDITIONAL_PYTHON_TIMES"
CMD+=" --additional_search_times $ADDITIONAL_SEARCH_TIMES"
CMD+=" --sample_timeout $SAMPLE_TIMEOUT"
CMD+=" --max_evolve_times $MAX_EVOLVE_TRAJECTORIES_TIMES"
if [ "$USE_LOG" = true ]; then
    CMD+=" --use_log"
fi
if [ "$USE_SDS" = true ]; then
    CMD+=" --use_sds"
fi

# If DATA_PATH is not empty, add the parameter
if [ ! -z "$DATA_PATH" ]; then
    CMD+=" --data_path $DATA_PATH"
fi

# Add tool config parameters
CMD+=" --conda_env $CONDA_ENV"
CMD+=" --python_max_concurrent $PYTHON_MAX_CONCURRENT"
CMD+=" --bing_api_key $BING_API_KEY"
CMD+=" --bing_zone $BING_ZONE"
CMD+=" --search_max_results $SEARCH_MAX_RESULTS"
CMD+=" --search_result_length $SEARCH_RESULT_LENGTH"
CMD+=" --bing_requests_per_second $BING_REQUESTS_PER_SECOND"
CMD+=" --bing_max_retries $BING_MAX_RETRIES"
CMD+=" --bing_retry_delay $BING_RETRY_DELAY"

# Additional parameters for search tool
CMD+=" --summ_model_urls $SUMM_MODEL_URLS"
CMD+=" --summ_model_name $SUMM_MODEL_NAME"
CMD+=" --summ_model_path $SUMM_MODEL_PATH"
CMD+=" --search_cache_file $SEARCH_CACHE_FILE"
CMD+=" --url_cache_file $URL_CACHE_FILE"

if [ "$COMPATIBLE_SEARCH" = true ]; then
    CMD+=" --use_local_search"
    CMD+=" --local_search_url $LOCAL_SEARCH_URL"
    CMD+=" --compatible_search"
else
    if [ "$USE_LOCAL_SEARCH" = true ]; then
        CMD+=" --use_local_search"
        CMD+=" --local_search_url $LOCAL_SEARCH_URL"
    fi
fi

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

echo $CMD

# Execute command
eval $CMD | tee logs/infer.log 2>&1