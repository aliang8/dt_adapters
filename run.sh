CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=offline_finetune \
    model_ckpt_dir=/model/checkpoint/dir \
    env_name=pick-place-wall-v2

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    general.batch_size=256 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_block_only_no_images_10.hdf5 \
    general.exp_name=pretrain_9_block_tasks_100_demos \
    general.log_to_wandb=False \
    general.stage=finetuning \



CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 zero_shot_dt_eval.py \
    --config-name=eval \
    num_processes=5 \
    general.log_to_wandb=False \
    general.log_outputs=False \
    general.load_from_ckpt=True \
    general.model_ckpt_dir=/data/anthony/dt_adapters/outputs/pretrain_9_block_tasks_10_demos/exp-659011 \
    metrics_file=/data/anthony/dt_adapters/metrics_10.csv \
    general.obj_randomization=True


# =============================
# MLP experiments
# =============================

# Pretraining
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=train \
    data=[base,mw_45_5] \
    model=[base,mlp_policy] \
    general.batch_size=256 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_all_with_images_10.hdf5 \
    general.exp_name=pretrain_ml_45_images_mlp_policy \
    general.log_to_wandb=False \
    general.stage=pretraining \
    general.log_outputs=False \
    general.load_from_ckpt=False \
    general.use_adapters=False \
    general.eval_every=0 \
    data.hide_goal=True \
    model.state_encoder.num_layers=6 \
    model.num_prediction_head_layers=4 \
    model.hidden_size=256


# MLP fine-tune
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=offline_finetune \
    data=[base,mw_45_5] \
    model=[base,mlp_policy] \
    data.data_file=trajectories_all_no_images_10.hdf5 \
    general.exp_name=mlp_policy_mw45_hand_insert_ft_full \
    general.log_to_wandb=True \
    general.stage=finetuning \
    general.log_outputs=True \
    general.model_ckpt_dir=/data/anthony/dt_adapters/outputs/pretrain_ml_45_no_goals_mlp_policy_larger_w_relu/exp-658003 \
    general.use_adapters=False \
    general.eval_every=1 \
    general.obj_randomization=True \
    general.freeze_backbone=False \
    data.finetune_tasks=["hand-insert-v2"] \
    model.num_layers=6 \
    model.num_prediction_head_layers=4 \
    model.hidden_size=256

# =============================
# DT experiments
# =============================

# Finetune
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=offline_finetune \
    data=[base,mw_45_5] \
    model=[base,decision_transformer] \
    general.batch_size=128 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_all_no_images_10.hdf5 \
    general.exp_name=offline_finetune_bin-picking-v2_mw45_finetune_full \
    general.log_to_wandb=False \
    general.stage=finetuning \
    general.log_outputs=False \
    general.model_ckpt_dir=/data/anthony/dt_adapters/outputs/pretrain_ml_45_no_goals/exp-122292 \
    general.use_adapters=False \
    model.stochastic=False \
    general.eval_every=1 \
    general.obj_randomization=True \
    model.num_layers=3 \
    model.emb_state_separate=True \
    model.hidden_size=768 \
    data.finetune_tasks=["bin-picking-v2"]

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=offline_finetune \
    general.batch_size=128 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_all_no_images_10.hdf5 \
    general.exp_name=offline_finetune_hand-insert-v2_mw45_finetune_full  \
    general.log_to_wandb=True \
    general.stage=finetuning \
    general.log_outputs=True \
    general.model_ckpt_dir=/data/anthony/dt_adapters/outputs/pretrain_ml_45_no_goals/exp-122292 \
    general.use_adapters=False \
    model.stochastic=False \
    general.eval_every=1 \
    general.obj_randomization=True \
    data.finetune_tasks=["hand-insert-v2"]

# Pretrain DT
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=train \
    data=[base,mw_45_5] \
    model=[base,decision_transformer] \
    general.batch_size=256 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_all_no_images_10.hdf5 \
    general.exp_name=pretrain_ml_45_v2 \
    general.log_to_wandb=True \
    general.stage=pretraining \
    general.log_outputs=True \
    general.load_from_ckpt=False \
    general.use_adapters=False \
    model.stochastic=False \
    general.eval_every=0 \
    data.hide_goal=True \
    model.num_layers=3 \
    model.emb_state_separate=True \
    model.hidden_size=768 \


# Pretrain only data from single task
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=train \
    data=metaworld \
    general.batch_size=128 \
    general.num_epochs=100 \
    general.num_steps_per_epoch=200 \
    general.num_online_rollouts=1 \
    data.data_file=trajectories_all_no_images_10.hdf5 \
    general.exp_name=pretrain_scratch_all_block_tasks_ft_pick_place_wall \
    general.log_to_wandb=True \
    general.stage=pretraining \
    general.log_outputs=True \
    general.load_from_ckpt=False \
    general.use_adapters=False \
    model.stochastic=False \
    general.env_name=pick-place-wall-v2 \
    general.eval_every=1 \
    general.log_eval_videos=True \
    general.obj_randomization=True 