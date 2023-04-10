export WANDB_DISABLED=YES
torchrun --nproc_per_node=8 --master_port=12315 train.py \
    --bf16 True \
    --data_path alpaca_data.json \
    #--data_path alpaca_data_merge_persona_v0.2.json \
    --output_dir sft-persona-7b.1.0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --xpu_backend ccl