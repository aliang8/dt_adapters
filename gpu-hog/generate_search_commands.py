"""
Single task training! 
python3 generate_search_commands.py \
    --config_name finetune_single_adapter \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file sta_run.sh \
    --run_sta 1 \
    --exp_prefix sta \
    --task_set sta 
    
Single task training for fusion tasks!
python3 generate_search_commands.py \
    --config_name finetune_single_adapter \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_sta_run.sh \
    --run_sta 0 \
    --exp_prefix sta_fusion \
    --task_set fusion \
    --seeds 0

Bert-fusion training!
python3 generate_search_commands.py \
    --config_name finetune_fusion \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_run.sh \
    --exp_prefix bert_fusion \
    --task_set fusion \
    --fusion_method bert-fusion \
    --seeds 0 1 2

python3 generate_search_commands.py \
    --config_name finetune_fusion \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file fusion_weighted_comp_run.sh \
    --seeds 0 \
    --exp_prefix fusion_weighted_comp
    --task_set fusion \
    --fusion_method weighted-composition \

# Evaluation
python3 generate_search_commands.py \
    --config_name eval \
    --model_ckpt_dir /data/anthony/dt_adapters/results/pretraining_40_tasks_scheduler_2 \
    --output_script_file eval_sta.sh \
    --seeds 0 \
    --exp_prefix sta \
    --task_set sta \
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
        "--config_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--exp_prefix", default="sta"
    )  # sta or fusion or fusion_weighted_comp
    parser.add_argument(
        "--fusion_method", default="bert-fusion"
    )  # bert-fusion or weighted-composition
    parser.add_argument("--task_set", default="sta")  # sta or fusion
    parser.add_argument("--seeds", default=[0], type=int, nargs="+")

    args = parser.parse_args()
    print(vars(args))

    sta_exp_name = "sta"

    if args.task_set == "sta":
        tasks = st_adapter_tasks
    else:
        tasks = fusion_tasks

    with open(args.output_script_file, "w") as f:

        for seed in args.seeds:
            for task in tasks:
                command_args = {
                    "general.seed": int(seed),
                    "data.eval_task": task,
                    "general.exp_name": args.exp_prefix,
                    "general.model_ckpt_dir": args.model_ckpt_dir,
                }

                if args.config_name != "eval":
                    command_args.update(
                        {
                            "general.overwrite_folder": True,
                            "general.log_to_wandb": True,
                        }
                    )

                if "fusion" in args.config_name:
                    adapters_to_use = [
                        f"{adapter}_{sta_exp_name}_{int(seed)}"
                        for adapter in st_adapter_tasks
                    ]
                    adapters_to_use = "[" + ",".join(adapters_to_use) + "]"
                    command_args.update(
                        {
                            "model.adapter_keys_to_use": adapters_to_use,
                            "model.adapter_config.fusion.fusion_method": args.fusion_method,
                        }
                    )

                exp_name = f"{args.exp_prefix}_finetune_{task}"
                cmd = f"DISPLAY=:0 python3 ../dt_adapters/trainer.py --config-name={args.config_name}"
                for k, v in command_args.items():
                    cmd += f" {k}={v}"
                f.write(cmd + "\n")
