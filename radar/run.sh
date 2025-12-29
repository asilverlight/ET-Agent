endpoints=(
    "http://localhost:8031/v1"
)
model_name="Qwen2.5-72B-Instruct"
data_names=(
    "aime24"
    "amc23"
    "math500"
    "2wiki"
    "bamboogle"
    "musique"
)
methods_paths=(
    "/path/to/output"
)

exp_types=(
    "evaluate_redundancy"
    "evaluate_tool_execute_errors"
    "evaluate_thinking_length"
)

COUNTS=500

for exp_type in ${exp_types[@]}; do
    for data_name in ${data_names[@]}; do
        data_path="${methods_paths[0]}/${data_name}_output_${COUNTS}.json"
        output_path="${methods_paths[0]}/${data_name}_${exp_type}_output_${COUNTS}.json"
        CMD="python /path/to/radar/run.py"
        CMD+=" --data_path ${data_path}"
        CMD+=" --output_path ${output_path}"
        CMD+=" --default_model ${model_name}"
        CMD+=" --endpoints ${endpoints[@]}"
        CMD+=" --exp_type ${exp_type}"
        echo ${CMD}
        eval ${CMD}
    done
done