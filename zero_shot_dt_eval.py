"""
Script to measure offline pretraining performance
"""
import os
import csv
import glob
import wandb
import torch
import hydra
import numpy as np
import time
import collections
from pprint import pprint
import multiprocessing as mp
from collections import defaultdict as dd
from collections import Counter
from omegaconf import OmegaConf
from multiprocessing.managers import BaseManager

import mw_constants
import mw_utils
import eval_utils
import general_utils
from train_mw import Trainer, set_all_seeds


def zero_shot_eval(config, runs, results_queue):
    eval_rollouts = []
    start = time.time()
    for (env_name, seed) in runs:
        print(f"working on {env_name}, {seed}")
        trainer = Trainer(config)
        set_all_seeds(seed)

        env_name = env_name.replace("_", "-")
        assert "v2" in env_name

        for _ in range(config.num_eval_rollouts):
            # reset the environment
            trainer.setup_env(env_name)
            with torch.no_grad():
                path = trainer.rollout(
                    use_means=True, attend_to_rtg=False, log_eval_videos=False
                )

            if results_queue is not None:
                results_queue.put((env_name, seed, path))

            eval_rollouts.append(path)

        print(f"finished {env_name}, {seed} in {time.time() - start} secs")


def handle_output(config, results_queue, seeds):
    # stats = dd(lambda: dd(int))
    # videos = dd(list)
    paths = dd(lambda: dd(list))
    metrics = []

    while True:
        out = results_queue.get()
        if out is not None:
            env_name, seed, path = out
            paths[env_name][seed].append(path)

            # average results across seeds
            done_with_all_seeds = True

            for seed in seeds:
                if len(paths[env_name][seed]) != config.num_eval_rollouts:
                    done_with_all_seeds = False

            if done_with_all_seeds:
                print(f"done with all seeds for {env_name}")

                avg_metrics = dd(int)
                for seed in seeds:
                    path_metrics = eval_utils.compute_eval_metrics(
                        paths[env_name][seed]
                    )
                    for k, v in path_metrics.items():
                        avg_metrics[k] += v

                avg_metrics = {k: v / len(seeds) for k, v in avg_metrics.items()}
                pprint(avg_metrics)

                avg_metrics["env_name"] = env_name
                metrics.append(avg_metrics)

            # videos[env_name].append(path["env_infos"]["frames"])

            # if (
            #     len(videos[env_name]) == config.num_eval_rollouts
            #     and config.log_to_wandb
            # ):
            #     print(f"done with {env_name}, saving videos")
            #     start = time.time()
            #     # log video
            #     trainer.save_videos(videos[env_name])

            # success = np.any(path["env_infos"]["success"])
            # if success:
            #     stats[env_name]["success_rate"] += 1
        else:
            # write metrics to file
            keys = list(metrics[0].keys())
            with open(config.metrics_file, "w") as csv_file:
                dict_writer = csv.DictWriter(csv_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(metrics)

            break

    # if config.log_to_wandb:
    #     data = [
    #         [env, stats["success_rate"] / config.num_eval_rollouts]
    #         for env, stats in stats.items()
    #     ]
    #     table = wandb.Table(data=data, columns=["env_name", "sr"])
    # trainer.wandb_logger.log(
    #     {
    #         "success_rates": wandb.plot.bar(
    #             table, "env_name", "sr", title="Success Rates"
    #         )
    #     }
    # )


@hydra.main(config_path="configs", config_name="eval")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)

    envs_to_evaluate = []
    if config.filter_envs_by_obj:
        envs_to_evaluate = mw_constants.OBJECTS_TO_ENV[config.filter_envs_by_obj]
    else:
        envs_to_evaluate = ["reach-v2"]

    seeds = [0, 1, 2]

    runs = []
    for seed in seeds:
        for env_name in envs_to_evaluate:
            runs.append((env_name, seed))

    print(f"number of runs: {len(runs)}")

    if config.num_processes > 0:
        torch.multiprocessing.set_start_method("spawn")
        results_queue = mp.Queue()

        # BaseManager.register("Trainer", Trainer)
        # manager = BaseManager()
        # manager.start()
        # trainer = manager.Trainer(config)

        proc = mp.Process(
            target=handle_output,
            args=(config, results_queue, seeds),
        )

        processes = []
        proc.start()

        num_processes = min(len(runs), config.num_processes)

        runs = list(general_utils.split(runs, num_processes))

        for rank in range(num_processes):
            p = mp.Process(
                target=zero_shot_eval,
                args=(config, runs[rank], results_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results_queue.put(None)
        proc.join()
    else:
        zero_shot_eval(config, [runs[0]], None)


if __name__ == "__main__":
    main()
