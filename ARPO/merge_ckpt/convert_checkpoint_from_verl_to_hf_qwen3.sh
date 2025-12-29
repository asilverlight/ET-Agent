

# 使用空格分隔的数字列表
for i in 10 20 30; do

    checkpoint_dir="<ckpt_path>/global_step_${i}/actor"

    BASE_MODEL="<your_base_model_path>"

    target_dir="<your_target_model_path>/global_step_${i}/hf"

    echo "Processing step $i..."
    echo "Checkpoint dir: $checkpoint_dir"
    echo "Target dir: $target_dir"

    python3 <your_path_to_ARPO>/convert_checkpoint_from_verl_to_hf.py merge \
        --backend "fsdp" \
        --hf_model_path "$BASE_MODEL" \
        --local_dir "$checkpoint_dir" \
        --target_dir "$target_dir"

done


