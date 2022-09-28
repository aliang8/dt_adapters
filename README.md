# DT Adapters Project

Using decision transformers as the backbone for pretraining offline a behavior policy on demonstration / play data. We then fine-tune individual adapters on top of the pretrained model for each downstream skill. To accelerate learning of compositional skills, we leverage AdapterFusion to merge several previously learned adapters together. 

# Project Setup

Setup conda env:

```
conda create -n dt_adapters python=3.7.9
```

Install metaworld 

```
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld 
pip install -e .
```

Install requirements
```
pip install -r requirements.txt
```

Some exports needed for when running mujoco_py
```
export DATA_DIR=/home/anthony/dt_adapters/data
export LOG_DIR=/home/anthony/dt_adapters/outputs
mkdir -p ${LOG_DIR}/slurm_outputs
export WANDB_API_KEY=YOUR_WANDB_KEY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
``` 

Extra setup: installing robosuite, robomimic, etc

## Data collection
```
# Train RL policy with custom implementation (doesn't work)
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/train.py env=reach-wall-v2-goal-observable log_to_wandb=false

# Train SAC policies with garage (works well for single task no randomization)
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 -m garage.examples.torch.sac_metaworld --env_name=assembly-v2-goal-observable --seed=0 --gpu=0

# Eval SAC policies trained with garage
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/eval_garage.py demos_per_env=10 debug=true model=single_task_sac log_to_wandb=true

# Collect demos using scripted policies
# TODO: add argparsing here
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 collect_scripted_policy_demos.py
```

## Pretraining DT model 
```
# Training
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    general.batch_size=32 \
    data.data_file=trajectories_block_only_no_images_10.hdf5 \
    general.exp_name=test \
    general.log_to_wandb=false

# Zero-shot inference
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 zero_shot_dt_eval.py \
    num_processes=0 \
    log_to_wandb=false
```

## Finetuning on adapters
```
# Online training 
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py \
    --config-name=online_finetune \
    model_ckpt_dir=/model/checkpoint/dir \
    env_name=pick-place-wall-v2
```

## Running grid search experiments
```
# running on the cluster
python3 slurm_jobber.py \
    --num_processes_per_gpu=1 \
    --run_scripts=1 \
    --mode=online \
    --grid_files=experiments/exp_obj_randomization.json \
    --node=ron \
    --lower_priority=1

python3 slurm_jobber.py \
    --num_processes_per_gpu=3 \
    --run_scripts=0 \
    --mode=online \
    --grid_files=experiments/exp_adapter_vs_no_adapter.json \
    --run_amber
```

## Notes
Current machines that run mujoco: lucy, ellie, ron, titan
Amber