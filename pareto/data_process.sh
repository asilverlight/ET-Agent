python /path/to/pareto/data_process/data_process.py \
    --data_path /path/to/output.json \
    --output_path /path/to/output_pareto.json \
    --counts 8000 \
    --exp_type process_pareto_data

python /path/to/pareto/data_process/data_process.py \
    --data_path /path/to/output_pareto.json \
    --output_path /path/to/ARPO/rl_datasets/train_processed_stage.parquet \
    --exp_type make_parquet_datas