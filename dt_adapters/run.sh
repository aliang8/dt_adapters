# =============================
# Eval
# =============================

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters.trainer \
    --config-name=eval \
    num_processes=5 \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.load_from_ckpt=True \
    general.model_ckpt_dir=/data/anthony/dt_adapters/outputs/pretrain_9_block_tasks_10_demos/exp-659011 \
    metrics_file=/data/anthony/dt_adapters/metrics_10.csv \
    general.obj_randomization=True


# =============================
# MLP policy experiments
# =============================

# Pretraining MLP model using only state input
CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=train \
    data=[base,mw_45_5] \
    model=[base,mlp_policy] \
    general.exp_name=pretrain_ml45_images_mlp \
    general.log_to_wandb=True \
    general.log_outputs=True \
    general.eval_every=0 \
    data.hide_goal=True \
    model.state_encoder.num_ll_enc_layers=6 \
    model.num_prediction_head_layers=4 \
    model.hidden_size=256 \
    data.state_keys=low_level # change this for state + vision input [low_level,image] or just image


# MLP fine-tune prediction head on some downstream task
# change freeze_backbone if want to fine-tune the entire model
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=offline_finetune \
    data=[base,mw_45_5] \
    model=[base,mlp_policy] \
    general.exp_name=mlp_ml45_hand_insert_adapter \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.load_from_ckpt=True \
    general.model_ckpt_dir=/path/to/model/ckpt \
    general.eval_every=1 \
    general.freeze_backbone=True \
    model.use_adapters=False \
    data.obj_randomization=True \
    data.eval_task=hand-insert-v2 \

# ========================================
# Decision Transformer policy experiments
# ========================================

# Pretrain DT on Metaworld
CUDA_VISIBLE_DEVICES=3 DISPLAY=:3 python3 -m dt_adapters.trainer \
    --config-name=train \
    data=[base,mw_45_5] \
    model=[base,decision_transformer] \
    general.exp_name=pretrain_ml45_images_dt \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.eval_every=0 \
    data.hide_goal=True \
    model.state_encoder.num_ll_enc_layers=6 \
    data.state_keys=low_level

# Train DT with adapters on some downstream task
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=offline_finetune \
    data=[base,mw_45_5] \
    model=[base,decision_transformer] \
    general.exp_name=dt_ml45_bin_picking_adapter \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.model_ckpt_dir=/path/to/model/ckpt \
    general.eval_every=1 \
    general.obj_randomization=True \
    data.eval_task=bin-picking-v2

# Fine-tune full model on downstream task data
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=offline_finetune \
    data=[base,mw_45_5] \
    model=[base,decision_transformer] \
    general.exp_name=offline_finetune_bin-picking-v2_mw45_finetune_full \
    general.log_to_wandb=False \
    general.log_outputs=False \
    model.use_adapters=False \
    general.model_ckpt_dir=/path/to/model/ckpt \
    general.eval_every=1 \
    general.obj_randomization=True \
    data.eval_task=bin-picking-v2

# Pretrain DT on RLBench
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=train \
    data=[base,rlbench] \
    model=[base,decision_transformer] \
    general.exp_name=pretrain_ml45_images_dt \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.eval_every=0 \
    data.hide_goal=True \
    data.state_keys=[image,low_level]
    model.state_encoder.num_ll_enc_layers=6 