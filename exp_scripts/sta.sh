export WANDB_API_KEY=c3346316614a2387db57382fe38090bd4c011d30
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tejas/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

DISPLAY=:0 /home/tejas/anaconda3/envs/dt_ada/bin/python3 dt_adapters/trainer.py \
    --config-name=finetune_single_adapter \
    data.eval_task=${1} \
    general.exp_name=test \
    general.model_ckpt_dir=/home/jaiv/dt_adapters/dt_adapters/results/pretraining_cw/1 \
    general.log_to_wandb=true