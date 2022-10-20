# Train DT with adapters on some downstream task
CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=offline_finetune \
    data=[base,rlbench] \
    model=[base,decision_transformer] \
    general.exp_name=adapter_rlbench_10tasks \
    general.log_to_wandb=True \
    general.log_outputs=True \
    general.model_ckpt_dir=/data/ishika/dt_adapters/outputs/pretrain_rlbench_10tasks/exp-720686 \
    general.eval_every=5 \
    general.freeze_backbone=True \
    data.obj_randomization=True \
    data.eval_task=open_box \
    data.observation_mode=image \
    general.num_processes=0 &

CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python3 -m dt_adapters.trainer \
    --config-name=offline_finetune \
    data=[base,rlbench] \
    model=[base,decision_transformer] \
    general.exp_name=adapter_rlbench_10tasks \
    general.log_to_wandb=True \
    general.log_outputs=True \
    general.model_ckpt_dir=/data/ishika/dt_adapters/outputs/pretrain_rlbench_10tasks/exp-720686 \
    general.eval_every=5 \
    general.freeze_backbone=True \
    data.obj_randomization=True \
    data.eval_task=close_microwave \
    data.observation_mode=image_state \
    general.num_processes=2 \
    general.skip_first_eval=False
    
    # \
    # general.num_eval_rollouts=10 \
    # data.max_ep_len=2 &

# CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python3 -m dt_adapters.trainer \
#     --config-name=offline_finetune \
#     data=[base,rlbench] \
#     model=[base,decision_transformer] \
#     general.exp_name=adapter_rlbench_10tasks \
#     general.log_to_wandb=True \
#     general.log_outputs=True \
#     general.model_ckpt_dir=/data/ishika/dt_adapters/outputs/pretrain_rlbench_10tasks/exp-720686 \
#     general.eval_every=5 \
#     general.freeze_backbone=True \
#     data.obj_randomization=True \
#     data.eval_task=unplug_charger \
#     data.observation_mode=image \
#     general.num_processes=0 &

# CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python3 -m dt_adapters.trainer \
#     --config-name=offline_finetune \
#     data=[base,rlbench] \
#     model=[base,decision_transformer] \
#     general.exp_name=adapter_rlbench_10tasks \
#     general.log_to_wandb=True \
#     general.log_outputs=True \
#     general.model_ckpt_dir=/data/ishika/dt_adapters/outputs/pretrain_rlbench_10tasks/exp-720686 \
#     general.eval_every=5 \
#     general.freeze_backbone=True \
#     data.obj_randomization=True \
#     data.eval_task=toilet_seat_up \
#     data.observation_mode=image \
#     general.num_processes=0 &

# CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python3 -m dt_adapters.trainer \
#     --config-name=offline_finetune \
#     data=[base,rlbench] \
#     model=[base,decision_transformer] \
#     general.exp_name=adapter_rlbench_10tasks \
#     general.log_to_wandb=True \
#     general.log_outputs=True \
#     general.model_ckpt_dir=/data/ishika/dt_adapters/outputs/pretrain_rlbench_10tasks/exp-720686 \
#     general.eval_every=5 \
#     general.freeze_backbone=True \
#     data.obj_randomization=True \
#     data.eval_task=open_fridge \
#     data.observation_mode=image \
#     general.num_processes=0 &