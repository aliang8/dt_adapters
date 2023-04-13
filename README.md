# DT Adapters Project

Using decision transformers as the backbone for pretraining offline a behavior policy on demonstration / play data. We then fine-tune individual adapters on top of the pretrained model for each downstream skill. To accelerate learning of compositional skills, we leverage AdapterFusion to merge several previously learned adapters together. 

# Project Setup

Setup conda env:
```
conda create -prefix ./dt_adapters python=3.8
```

Clone this repo. 
```
git clone --recursive https://github.com/aliang8/dt_adapters/tree/simplified/dt_adapters
git checkout simplified
git submodule update --init --recursive
git pull --recurse-submodules
```

Install requirements
```
pip install -r requirements.txt
```

Some exports needed for when running mujoco_py
```
export WANDB_API_KEY=YOUR_WANDB_KEY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
``` 

## Data collection
```
# Collect demos using scripted policies for all tasks 
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/data/collect_scripted_policy_demos.py \
    --data_dir /data/anthony/dt_adapters/data \
    --num_demos 25

# Collect demos using scripted policies for a specific task
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/data/collect_scripted_policy_demos.py \
    --task_name pick-place-v2 \
    --data_dir /data/anthony/dt_adapters/data \
    --num_demos 10
```

## Pretraining and fine-tuning transformer policy
```
# Pretraining
# Note: don't do any eval during pretraining
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=pretrain \
    general.eval_every=0 \
    general.exp_name=pretraining \
    general.log_to_wandb=true

CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=pretrain \
    general.eval_every=0 \
    general.exp_name=pretraining_40_tasks_scheduler_2 \
    general.num_epochs=500 \
    general.log_to_wandb=true \
    general.use_lr_scheduler=true \
    general.load_from_ckpt=True \
    general.resume_experiment=True \
    general.model_ckpt_dir=/data/anthony/dt_adapters/results/pretraining_40_tasks_2 \

# Fine-tuning single task adapter for new downstream task
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=finetune_single_adapter \
    data.eval_task=pick-place-v2 \
    general.exp_name=test \
    general.model_ckpt_dir=/data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \

# Fine-tuning fusion layer for new downstream task
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=finetune_fusion \
    data.eval_task=bin-picking-v2 \
    general.exp_name=finetune_fusion_test \
    general.model_ckpt_dir=/data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    model.adapters_to_use=[pick-place-v2,door-open-v2]


# Eval trained model
CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=eval \
    data.eval_task=pick-place-v2 \
    general.exp_name=sta \
    general.model_ckpt_dir=/data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    model.use_single_adapter=True \
```

```
Structure of adapter_library:


```


```
Run with gpu-hog for parallelizing multiple jobs

python3 main.py --job_file fusion_run.sh --gpus 1,1,1,1,1
```

## Notes
Current machines that run mujoco: lucy, ellie, ron, titan
Amber

To kill job on certain GPU (0)
```
kill $(nvidia-smi -g 0 | awk '$5=="PID" {p=1} p {print $5}')
```