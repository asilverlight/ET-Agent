# conda activate /path/to/conda/bin/python
module load cuda/12.1.1
cd /path/to/LlaMA-Factory
CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train /path/to/LlaMA-Factory/examples/train_full/train_sft.yaml


