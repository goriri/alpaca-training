export WANDB_DISABLED=YES
deepspeed --include localhost:0,1,3,5 train.py \
    --model_name_or_path "/data1/nlp/baiy/data/llama/7B-2hf/llama-7b" \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir /data1/nlp/baiy/output-model/alpaca-ds-llama-b4-gt32-e3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed conf/ds_llama7b_config2.json
