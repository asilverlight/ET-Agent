module load cuda/12.1.1
# CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwen2_lora_sft.yaml
# CUDA_VISIBLE_DEVICES=0,2 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwen2_lora_simpo.yaml
cd /home/u2024001021/LLaMA-Factory-main
# CUDA_VISIBLE_DEVICES=5,6 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwen2_lora_kto.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwq_lora_dpo.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwq_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwq_lora_dpo.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /home/u2024001021/LLaMA-Factory-main/examples/train_lora/qwq_lora_dpo_2.yaml