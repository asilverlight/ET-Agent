#!/bin/bash

# 定义任务映射表：将数据集名称映射到对应的任务类型
# 使用关联数组存储数据集名称和任务类型的对应关系
declare -A task_map=(
    ["aime24"]="math"      # AIME24数学竞赛数据集
    ["aime25"]="math"      # AIME25数学竞赛数据集
    ["amc23"]="math"       # AMC23数学竞赛数据集
    ["math500"]="math"     # 500道数学题数据集
    ["olympiadbench"]="math" # OlympiadBench数学竞赛数据集
    ["gsm8k"]="math"        # GSM8K数学数据集
    ["math"]="math"        # 通用数学数据集
    ["webwalker"]="qa"     # 网页问答数据集
    ["hotpotqa"]="qa"      # HotpotQA多跳问答数据集
    ["2wiki"]="qa"         # 2Wiki多跳问答数据集
    ["bamboogle"]="qa"     # Bamboogle搜索问答数据集
    ["musique"]="qa"       # Musique多跳问答数据集
    ["hle"]="qa"           # HLE问答数据集
    ["gaia"]="qa"          # GAIA问答数据集
    ["csbench"]="qa"       # CSBench问答数据集
    ["gpqa"]="qa"          # GPQA问答数据集
)

# 设置输出目录路径，用于存放需要评估的结果文件
OUTPUT_DIR=(
    "/path/to/output"
)

# 主循环：两层循环，第一层遍历OUTPUT_DIR，第二层遍历所有数据集名称

COUNTS=500

for dir in "${OUTPUT_DIR[@]}"; do
    for dataset_name in "${!task_map[@]}"; do
        # 构造文件路径，格式为：$dir/$dataset_name.json
        file_path="$dir/${dataset_name}_output_${COUNTS}.json"

        # 检查文件是否存在
        if [ ! -f "$file_path" ]; then
            # 文件不存在，跳过
            echo "File not found: $file_path"
            continue
        fi

        # 获取任务类型
        task="${task_map[$dataset_name]}"

        # 检查是否找到了对应的任务类型
        if [ -z "$task" ]; then
            echo "Unknown dataset: $dataset_name. Skipping..."  # 未知数据集，跳过处理
            continue
        fi

        # 输出当前正在评估的数据集信息
        echo "Evaluating $dataset_name (task: $task) in $dir..."

        # 调用评估脚本，设置环境变量：
        # OUTPUT_PATH: 指定要评估的文件路径
        # TASK: 指定任务类型
        OUTPUT_PATH="$file_path" TASK="$task" USE_LLM=true COUNTS="$COUNTS" bash /path/to/evaluate.sh

        # 输出评估完成信息
        echo "Finished evaluating $dataset_name in $dir"
        echo "-----------------------------------"  # 分隔线
    done
done
