"""
Script for creating slurm files and then executing them for searching over hyperparamters
"""

import os
import glob
from sklearn.model_selection import ParameterGrid
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


PROGRAMS_PER_JOB = 1

# header for slurm, need to edit this
HEADER = """#!/bin/bash 
#SBATCH --job-name=train_dt_adapters
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=2080:1
#SBATCH --nodelist=ink-ellie
#SBATCH --output=/home/anthony/dt_adapters/slurm_output/%j.out

HOME=/home/anthony

# conda activate robomimic_venv

wandb login 0815350e6c514d36864729063abb10fc03898c00
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
export TOKENIZERS_PARALLELISM=false 
"""

BASE_CMD = "CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py "

# BASE_CMD = (
#     "CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python -m garage.examples.torch.sac_metaworld "
# )


# create slurm files
os.makedirs("slurm_files", exist_ok=True)

# create param grid
envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())

configs = []

experiment_grids = [
    {
        "data.context_len": [30, 50, 100],
        "log_to_wandb": ["true"],
        "exp_name": ["train_dt_offline_varying_context_len"],
    },
    {
        "data_file": [
            "trajectories_block_only_no_images_10.hdf5",
            "trajectories_block_only_no_images_50.hdf5",
        ],
        "log_to_wandb": ["true"],
        "exp_name": ["train_dt_offline_dataset_size"],
    },
    {
        "data_file": [
            "trajectories_block_only_no_images_10.hdf5",
            "trajectories_block_only_no_images_50.hdf5",
        ],
        "model.n_layer": [6],
        "model.n_head": [6],
        "batch_size": [32],
        "log_to_wandb": ["true"],
        "exp_name": ["train_dt_offline_model_size"],
    },
]
for grid in experiment_grids:
    config = list(ParameterGrid(grid))
    configs.extend(config)


# create individual slurm files
# chunk configs
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


chunks = list(chunks(configs, PROGRAMS_PER_JOB))

for i, chunk in enumerate(chunks):
    slurm_cmd = HEADER
    for j, cfg in enumerate(chunk):
        slurm_cmd += BASE_CMD
        key = ""
        for k, v in cfg.items():
            slurm_cmd += f"{k}={v} "
            key += f"_{k}_{v}"

        if j != len(chunk) - 1:
            slurm_cmd += "&\n"

    slurm_cmd = slurm_cmd[:-1]  # remove last space
    # print(slurm_cmd)
    file = os.path.join("slurm_files", f"run_train_{i}.slrm")
    with open(file, "w") as f:
        f.write(slurm_cmd)

    print(f"running: sbatch {file}")
    os.system(f"sbatch {file}")
