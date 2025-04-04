#!/bin/bash
NAME=$1

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port 25031 \
    train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --version qwen_1_5 \
    --data_path ${ANNOTATION_PATH} \
    --image_folder ${VIDEO_DIR} \
    --vision_tower "convnext-576" \
    --mm_projector_type mlp2x_gelu \
    --fps 4 \
    --min_fast_frames 16 \
    --video_frames 64 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tile_image_input False \
    --cross_attn_every_n_layers 8 \
    --cross_attn_implementation text-only-vanilla-flashattn \
    --cross_attn_gating_type whole-dynamic-tanh-warmup \
    --initialize_cross_attn_kv_from_lm True \
    --mm_video_pooling_stride 2 \
    --fast_token_spatial_stride 1 \
    --fast_token_temporal_stride 4 \
    --fast_token_temporal_sampling_stride 1 \
    --bf16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --cross_attention_layer_lr 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${NAME}
