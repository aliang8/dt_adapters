"""
Script for creating slurm files and then executing them for searching over hyperparamters

# Amber
python3 -m dt_adapters.slurm_jobber \
    --num_processes_per_gpu=1 \
    --run_scripts=0 \
    --grid_files=dt_adapters/experiments/pretrain_dt.yaml \
    --lower_priority=0 \
    --data=mw_40_10_cl \
    --model=transformer \
    --config=train \
    --run_amber
"""

import os
import glob
import json
import random
import yaml
import dt_adapters.general_utils as general_utils
from sklearn.model_selection import ParameterGrid

NODE_GPU_MAP = {
    "ellie": "2080",
    "lucy": "1080",
    "ron": "1080",
    "titan": "6000",
    "gary": "2080",
}
NUM_GPUS_AVAILABLE = 4

import collections


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
export LOG_DIR={os.environ["LOG_DIR"]}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{os.environ['USER']}/.mujoco/mujoco210/bin
export TOKENIZERS_PARALLELISM=false 
export QT_LOGGING_RULES='*.debug=false;qt.qpa.*=false'
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
        grid = flatten(grid, sep=".")
        config = list(ParameterGrid(grid))
        configs.extend(config)

    # create individual slurm files
    chunks = list(general_utils.chunks(configs, args.num_processes_per_gpu))

    base_cmd = f"CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3.7 -m dt_adapters.trainer --config-name={args.config} data=[base,{args.data}] model=[base,{args.model}] "

    for i, chunk in enumerate(chunks):
        if args.run_amber:
            slurm_cmd = ""
        else:
            slurm_cmd = header

        for j, cfg in enumerate(chunk):
            slurm_cmd += base_cmd
            key = ""
            for k, v in cfg.items():
                slurm_cmd += f"{k}={v} "
                key += f"_{k}_{v}"

            if args.run_amber:
                slurm_cmd += f"&> outputs/slurm_outputs/stdout_{random.randint(int(1e5), int(1e6) - 1)}.txt "

            if j != len(chunk) - 1:
                slurm_cmd += "&\n"

        slurm_cmd = slurm_cmd[
            :-1
        ]  # remove last space, important or the job will crash :/

        if args.run_amber:
            print(
                slurm_cmd.replace("CUDA_VISIBLE_DEVICES=0", f"CUDA_VISIBLE_DEVICES={i}")
                + " &"
            )
            print("=" * 50)
            if args.run_scripts:
                # redirect outputs to some file
                os.system(
                    slurm_cmd.replace(
                        "CUDA_VISIBLE_DEVICES=0", f"CUDA_VISIBLE_DEVICES={i}"
                    )
                    + " &"
                )
        else:
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
        "--node",
        type=str,
        default="ellie",
        help="which node on the nlp cluster to use",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ml45_5",
        help="data config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="decision_transformer",
        help="model config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="which yaml file to use for base",
    )
    parser.add_argument(
        "--grid_files",
        type=str,
        nargs="+",
        default="experiments/exp.json",
        help="path to json files that contain dictionary of parameters",
    )
    parser.add_argument(
        "--run_amber", action="store_true", help="are we running on nonslurm"
    )
    args = parser.parse_args()

    main(args)
