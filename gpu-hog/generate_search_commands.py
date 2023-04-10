"""
python3 generate_search_commands.py \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file sta_run.sh \
    --run_sta 1 \
    --exp_prefix sta
    
python3 generate_search_commands.py \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_sta_run.sh \
    --run_sta 0 \
    --exp_prefix sta

python3 generate_search_commands.py \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_run.sh \
    --fusion 1 \
    --exp_prefix fusion \
    --seeds 0 1 2
    
python3 generate_search_commands.py \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_weighted_comp_run.sh \
    --fusion 1 \
    --exp_prefix fusion_weighted_comp
"""

st_adapter_tasks = [
    "pick-place-v2",
    "door-open-v2",
    "faucet-open-v2",
    "handle-press-v2",
    "stick-pull-v2",
]

fusion_tasks = [
    "bin-picking-v2",
    "peg-insert-side-v2",
    "reach-wall-v2",
    "coffee-push-v2",
    "basketball-v2",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="generate commands for running experiments in parallel"
    )
    parser.add_argument(
        "--model_ckpt_dir",
        default="/data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2",
    )
    parser.add_argument(
        "--output_script_file",
        default="sta_run.sh",
    )
    parser.add_argument(
        "--run_sta",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fusion",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--exp_prefix", default="sta"
    )  # sta or fusion or fusion_weighted_comp
    parser.add_argument("--seeds", default=[0], type=int, nargs="+")

    args = parser.parse_args()
    print(vars(args))

    if args.fusion:  # run fusion experiment
        config_name = "finetune_fusion"
        tasks = fusion_tasks
    else:  # run single task adapter training
        config_name = "finetune_single_adapter"

        if args.run_sta:
            tasks = st_adapter_tasks
        else:
            tasks = fusion_tasks

    with open(args.output_script_file, "w") as f:

        for seed in args.seeds:
            for task in tasks:
                command_args = {
                    # "config-name": "finetune",
                    "general.seed": int(seed),
                    "data.eval_task": task,
                    "general.exp_name": f"{args.exp_prefix}_{task}_s_{int(seed)}",
                    "general.model_ckpt_dir": args.model_ckpt_dir,
                    "general.overwrite_folder": True,
                    "general.log_to_wandb": True,
                    "model.adapters_to_use": "[" + ",".join(st_adapter_tasks) + "]",
                    # "model.adapter_config.fusion.fusion_method": "weighted-composition",
                }

                exp_name = f"{args.exp_prefix}_finetune_{task}"
                cmd = f"DISPLAY=:0 python3 ../dt_adapters/trainer.py --config-name={config_name}"
                for k, v in command_args.items():
                    cmd += f" {k}={v}"
                f.write(cmd + "\n")
