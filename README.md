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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
``` 

Extra setup: installing robosuite, robomimic, etc

Train RL policy
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/train.py env=reach-wall-v2-goal-observable log_to_wandb=false
```

Evaluate trained RL policies
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/eval.py num_rollouts=10 env=window-open-v2-goal-observable restore_from_checkpoint=true
```

Collect trajectories for all metaworld env
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/save_dataset.py experiment=mw-dc-5M-t2 demos_per_env=10
```

Train Decision Transformer model using BC 
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py --dataset=/home/anthony/dt_adapters/pytorch_sac/data/trajectories.hdf5 --algo=dt --name=test


CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py
```

Train SAC policies with garage
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 -m garage.examples.torch.sac_metaworld --env_name=assembly-v2-goal-observable --seed=0 --gpu=0
```

Eval SAC policies trained with garage
```
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 pytorch_sac/eval_garage.py demos_per_env=10 debug=true model=single_task_sac log_to_wandb=true
```

# TODO
[ ] Collect demonstrations for metaworld environments