"""
Script for creating slurm files and then executing them for searching over hyperparamters
"""

import os
import glob
import json
import yaml
import general_utils
from sklearn.model_selection import ParameterGrid

NODE_GPU_MAP = {"ellie": "2080", "lucy": "1080", "ron": "1080", "titan": "6000"}


def main(args):
    # create slurm files
    os.makedirs("slurm_files", exist_ok=True)

    if args.lower_priority:
        more_gpus = "#SBATCH --qos=general"
    else:
        more_gpus = """
        """
    header = f"""#!/bin/bash
#SBATCH --job-name=train_dt_adapters
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node={NODE_GPU_MAP[args.node]}:1
#SBATCH --nodelist=ink-{args.node}
#SBATCH --output={os.environ['LOG_DIR']}/slurm_output/%j.out
{more_gpus}
HOME=/home/{os.environ['USER']}

wandb login {os.environ["WANDB_API_KEY"]}
export DATA_DIR=/home/anthony/dt_adapters/data
export LOG_DIR=/home/anthony/dt_adapters/outputs
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{os.environ['USER']}/.mujoco/mujoco210/bin
export TOKENIZERS_PARALLELISM=false 
"""

    configs = []
    for grid_file in args.grid_files:
        print(grid_file)
        # grid = json.load(open(grid_file, "r"))
        with open(grid_file, "r") as f:
            try:
                grid = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        print(grid)
        config = list(ParameterGrid(grid))
        configs.extend(config)

    # create individual slurm files
    chunks = list(general_utils.chunks(configs, args.num_processes_per_gpu))

    if args.mode == "pretraining":
        base_cmd = (
            "CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py --config-name=train "
        )
    elif args.mode == "online":
        base_cmd = "CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train_mw.py --config-name=online_finetune online_training=True "

    for i, chunk in enumerate(chunks):
        slurm_cmd = header

        for j, cfg in enumerate(chunk):
            slurm_cmd += base_cmd
            key = ""
            for k, v in cfg.items():
                slurm_cmd += f"{k}={v} "
                key += f"_{k}_{v}"

            if j != len(chunk) - 1:
                slurm_cmd += "&\n"

        slurm_cmd = slurm_cmd[
            :-1
        ]  # remove last space, important or the job will crash :/

        slrm_file = os.path.join("slurm_files", f"run_train_{i}.slrm")
        with open(slrm_file, "w") as f:
            f.write(slurm_cmd)

        if args.run_scripts:
            print(f"running: sbatch {slrm_file}")
            os.system(f"sbatch {slrm_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_processes_per_gpu",
        type=int,
        default=1,
        help="number of jobs that can fit on a single gpu",
    )
    parser.add_argument(
        "--run_scripts",
        type=int,
        default=0,
        help="should it also run sbatch or just create the files",
    )
    parser.add_argument(
        "--lower_priority",
        type=int,
        default=0,
        help="run the jobs with lower priority so i can take up more gpu space",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretraining",
        help="pretraining | online",
    )
    parser.add_argument(
        "--node",
        type=str,
        default="ellie",
        help="which node on the nlp cluster to use",
    )
    parser.add_argument(
        "--grid_files",
        type=str,
        nargs="+",
        default="experiments/exp.json",
        help="path to json files that contain dictionary of parameters",
    )
    args = parser.parse_args()

    main(args)
