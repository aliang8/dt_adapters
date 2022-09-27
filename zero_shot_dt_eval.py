import os
import glob
import wandb
import torch
import hydra
import numpy as np
import time
from general_utils import split
import collections
from collections import defaultdict as dd
import multiprocessing as mp
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

import mw_constants
import mw_utils
import eval_utils
from pprint import pprint
from train_mw import Trainer


def zero_shot_eval(config, runs, results_queue):
    print("initializing trainer...")
    trainer = Trainer(config)

    eval_rollouts = []
    for env_name in runs:
        env_name = env_name.replace("_", "-")
        assert "v2" in env_name

        for _ in range(config.num_eval_rollouts):
            # reset the environment
            trainer.setup_env(env_name)
            with torch.no_grad():
                path = trainer.rollout(
                    use_means=True, attend_to_rtg=False, phase="eval"
                )

            if results_queue is not None:
                results_queue.put((env_name, path))

            eval_rollouts.append(path)

        print(env_name)
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        pprint(metrics)


def handle_output(config, results_queue):
    stats = dd(lambda: dd(int))
    videos = dd(list)

    while True:
        out = results_queue.get()
        if out is not None:
            env_name, path = out
            videos[env_name].append(path["env_infos"]["frames"])

            if (
                len(videos[env_name]) == config.num_eval_rollouts
                and config.log_to_wandb
            ):
                print(f"done with {env_name}, saving videos")
                start = time.time()
                # log video
                trainer.save_videos(videos[env_name])

            success = np.any(path["env_infos"]["success"])
            if success:
                stats[env_name]["success_rate"] += 1
        else:
            break

    if config.log_to_wandb:
        data = [
            [env, stats["success_rate"] / config.num_eval_rollouts]
            for env, stats in stats.items()
        ]
        table = wandb.Table(data=data, columns=["env_name", "sr"])
        # trainer.wandb_logger.log(
        #     {
        #         "success_rates": wandb.plot.bar(
        #             table, "env_name", "sr", title="Success Rates"
        #         )
        #     }
        # )


@hydra.main(config_path="configs", config_name="eval")
def main(config):
    runs = []
    envs_to_evaluate = []
    if config.filter_envs_by_obj:
        envs_to_evaluate = mw_constants.OBJECTS_TO_ENV[config.filter_envs_by_obj]
    else:
        envs_to_evaluate = ["reach-v2"]

    # for e in range(config.num_eval_rollouts):
    #     for env in envs_to_evaluate:
    #         runs.append(env)
    runs = envs_to_evaluate

    print(f"number of runs: {len(runs)}")

    if config.num_processes > 0:
        torch.multiprocessing.set_start_method("spawn")
        results_queue = mp.Queue()

        proc = mp.Process(
            target=handle_output,
            args=(config, results_queue),
        )

        processes = []
        proc.start()

        runs = list(split(runs, config.num_processes))

        for rank in range(config.num_processes):
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
        zero_shot_eval(config, runs, None)


if __name__ == "__main__":
    main()
