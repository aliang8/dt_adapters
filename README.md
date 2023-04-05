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
# Collect demos using scripted policies
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/data/collect_scripted_policy_demos.py \
    --task_name pick-place-v2 \
    --data_dir /data/anthony/dt_adapters/data \
    --num_demos 10
```

## Pretraining DT model 
```
# Training
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/trainer.py \
    --config-name=base \
    general.eval_every=10 \
    general.exp_name=test \
    general.num_eval_rollouts=2 \
    general.log_to_wandb=false
```

## Notes
Current machines that run mujoco: lucy, ellie, ron, titan
Amber

To kill job on certain GPU (0)
```
kill $(nvidia-smi -g 0 | awk '$5=="PID" {p=1} p {print $5}')
```