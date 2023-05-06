export WANDB_API_KEY=c3346316614a2387db57382fe38090bd4c011d30
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tejas/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

DISPLAY=:0 /home/tejas/anaconda3/envs/dt_ada/bin/python3 dt_adapters/trainer.py \
    --config-name=finetune_fusion \
    data.eval_task=push-back-v2 \
    data.num_demos_per_task=${1} \
    model.adapter_config.fusion.fusion_method=taco-fusion \
    model.adapter_config.fusion.add_new_unfrozen_adapter=True \
    general.overwrite_folder=True \
    general.model_ckpt_dir=/home/jaiv/dt_adapters/dt_adapters/results/pretraining_cw/1 \
    general.exp_name=tacofusion_2existingtasks_demo${1} \
    model.adapter_keys_to_use=[push-v2_test_1,push-wall-v2_test_1] \
    general.skip_first_eval=True \
    general.log_to_wandb=True 