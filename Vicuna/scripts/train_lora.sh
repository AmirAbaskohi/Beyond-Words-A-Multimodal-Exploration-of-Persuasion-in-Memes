deepspeed fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /bigdata/amirhossein/FastChat/SemEval/vicuna.json \
    --output_dir ./checkpoints/vicuna-7b-v1.5-SemEval-lora_2e-4_new-caption_2epoch \
    --num_train_epochs 2 \
    --fp16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 51 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed /bigdata/amirhossein/FastChat/playground/deepspeed_config_s3.json \
    --gradient_checkpointing True \
    --flash_attn False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \