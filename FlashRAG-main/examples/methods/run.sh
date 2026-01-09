module load cuda/11.8
export PYTHONPATH=/home/u2024001049/FlashRAG-main:PYTHONPATH
# export CUDA_VISIBLE_DEVICES=6,7
# python /home/u2024001049/FlashRAG-main/examples/methods/run_exp.py \
#     --method_name selfrag \
#     --dataset_name arc \
#     --return_prob \
#     --gpu_id 1 \
#     --gpu_use 0.4 \
#     --dataset_path /home/u2024001049/share/feifei/datasets/metarag

# python /home/u2024001049/FlashRAG-main/examples/methods/run_exp.py \
#     --method_name adaptive \
#     --dataset_name arc \
#     --return_prob \
#     --gpu_id 1 \
#     --gpu_use 0.4 \
#     --dataset_path /home/u2024001049/share/feifei/datasets/metarag \
#     --generator_model llama3-8b-instruct

# python /home/u2024001049/FlashRAG-main/examples/methods/run_exp.py \
#     --method_name replug \
#     --dataset_name arc \
#     --return_prob \
#     --gpu_id 5 \
#     --gpu_use 0.3 \
#     --dataset_path /home/u2024001049/share/feifei/datasets/metarag \
#     --generator_model llama3-8b-instruct

# dataset_names=("triviaqa" "arc" "wikiasp")

# for dataset_name in "${dataset_names[@]}"; do
#     python /home/u2024001049/FlashRAG-main/examples/methods/run_exp.py \
#         --method_name iterretgen \
#         --dataset_name "$dataset_name" \
#         --return_prob \
#         --gpu_id 5 \
#         --gpu_use 0.3 \
#         --dataset_path /home/u2024001049/share/feifei/datasets/metarag \
#         --generator_model llama3-8b-instruct
# done

# 定义方法和数据集数组
methods=("flare") #
datasets=("2wiki" "wikiasp") #"arc"  "triviaqa"  "arc"

# # 固定参数
script_path="/home/u2024001049/FlashRAG-main/examples/methods/run_exp.py"
dataset_path="/home/u2024001049/share/feifei/datasets/metarag"
gpu_id=4
gpu_use=0.5

# 嵌套循环遍历所有组合
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running $method on $dataset..."
        python "$script_path" \
            --method_name "$method" \
            --dataset_name "$dataset" \
            --gpu_id "$gpu_id" \
            --gpu_use "$gpu_use" \
            --dataset_path "$dataset_path" \
            --generator_model qwen2.5-7b-instruct \
            --split source_retrieval_results
    done
done

# export CUDA_VISIBLE_DEVICES=4,5
# dataset_names=("triviaqa" "wikiasp" "arc") # 
# for dataset_name in "${dataset_names[@]}"
# do
#     python /home/u2024001049/MetaRAG/scripts/run.py \
#         --RAG_type agentic \
#         --dataset_name ${dataset_name} \
#         --gpu_use 0.7 \
#         --model_name llama3-8b-instruct \
#         --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/llama3-8b-instruct-agentic.json \
#         --max_model_len 8192 \
#         --return_prob \
#         --count 2,0
#     python /home/u2024001049/MetaRAG/scripts/evaluate.py \
#         --eval_type agentic \
#         --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/llama3-8b-instruct-agentic.json \
#         --dataset_name ${dataset_name} \
#         --model_name llama3-8b-instruct \
#         --result_log /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/result.log \
#         --additional agentic
# done

# export CUDA_VISIBLE_DEVICES=4,5
# dataset_names=("arc") # "2wiki""triviaqa" "wikiasp" 
# for dataset_name in "${dataset_names[@]}"
# do
#     # python /home/u2024001049/MetaRAG/scripts/run.py \
#     #     --RAG_type traj \
#     #     --dataset_name ${dataset_name} \
#     #     --gpu_use 0.7 \
#     #     --model_name llama3-8b-instruct \
#     #     --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/llama3-8b-instruct-traj-ensemble.json \
#     #     --max_model_len 8192 \
#     #     --return_prob \
#     #     --count 4,0
#     python /home/u2024001049/MetaRAG/scripts/evaluate.py \
#         --eval_type meta \
#         --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/llama3-8b-instruct-traj-ensemble.json \
#         --dataset_name ${dataset_name} \
#         --model_name llama3-8b-instruct \
#         --result_log /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/result.log \
#         --additional pipeline_ensemble,ensemble_model_llama3-8b-instruct_subsystem_replug_iter_retgen_selfrag_agentic \
#         --pipeline_ensemble
# done


# methods=("iterretgen" "replug") #"selfrag" 

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir /fs/archive/share/llama-2-7b-hf

# methods=("replug" "iterretgen")
# model_names=("llama3-8b-instruct" "qwen2-7b-instruct" "mistral-7b-instruct")
# 固定参数
# datasets=("computing" "film" "finance" "law" "music") # "biomedical"
# script_path="/home/u2024001049/FlashRAG-main/examples/methods/run_exp.py"
# dataset_path="/home/u2024001049/share/feifei/datasets/metarag/"
# gpu_id=5
# gpu_use=0.3

# # 嵌套循环遍历所有组合
# for dataset in "${datasets[@]}"; do
#     echo "Running llmlingua on $dataset..."
#     python "$script_path" \
#         --method_name llmlingua \
#         --dataset_name "$dataset" \
#         --gpu_id "$gpu_id" \
#         --gpu_use "$gpu_use" \
#         --dataset_path "$dataset_path" \
#         --generator_model qwen2.5-7b-instruct
# done

# export CUDA_VISIBLE_DEVICES=4,5
# dataset_names=("2wiki" "triviaqa" "wikiasp" "arc") # 
# for dataset_name in "${dataset_names[@]}"
# do
#     python /home/u2024001049/MetaRAG/scripts/run.py \
#         --RAG_type agentic \
#         --dataset_name ${dataset_name} \
#         --gpu_use 0.7 \
#         --model_name qwen2.5-7b-instruct \
#         --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/qwen2.5-7b-instruct-agentic.json \
#         --max_model_len 8192 \
#         --return_prob \
#         --count 2,0
#     python /home/u2024001049/MetaRAG/scripts/evaluate.py \
#         --eval_type agentic \
#         --result_path /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/qwen2.5-7b-instruct-agentic.json \
#         --dataset_name ${dataset_name} \
#         --model_name qwen2.5-7b-instruct \
#         --result_log /home/u2024001049/share/feifei/datasets/metarag/${dataset_name}/result.log \
#         --additional agentic
# done

