"""
Script for creating slurm files and then executing them for searching over hyperparamters
"""

import os
import glob
from sklearn.model_selection import ParameterGrid
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# header for slurm, need to edit this
HEADER = """#!/bin/bash 
#SBATCH --job-name=train_dt_adapters
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1080:1
#SBATCH --nodelist=ink-ron
#SBATCH --output=/home/anthony/dt_adapters/slurm_output/%j.out

HOME=/home/anthony

# conda activate robomimic_venv

wandb login 0815350e6c514d36864729063abb10fc03898c00
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco200/bin
export TOKENIZERS_PARALLELISM=false 
"""

# create slurm files
os.makedirs("slurm_files", exist_ok=True)

# create param grid
envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())[:3]
print(envs)
experiments = [f"mw-dc-{env}" for env in envs]

# param_grid = {"env": envs, "experiment": experiments, "log_to_wandb": ["true"]}
# configs = list(ParameterGrid(param_grid))
configs = []

for (env, experiment) in zip(envs, experiments):
    if os.path.exists(f"outputs/{experiment}"):
        print(f"skipping {experiment} because output file already exists")
    configs.append({"env": env, "experiment": experiment, "log_to_wandb": "true"})

# create individual slurm files
for cfg in configs:
    # print(cfg)
    slurm_cmd = HEADER + "CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train.py "

    key = ""
    for k, v in cfg.items():
        slurm_cmd += f"{k}={v} "
        key += f"_{k}_{v}"

    # print(slurm_cmd)
    with open(os.path.join("slurm_files", f"run_train{key}.slrm"), "w") as f:
        f.write(slurm_cmd)

# iterate through and run the files
files = glob.glob("slurm_files/*")
print(f"running {len(files)} slurm jobs")

for file in files:
    print(f"running: sbatch {file}")
    os.system(f"sbatch {file}")
