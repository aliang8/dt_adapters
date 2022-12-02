import os
import gc
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

import gym

from torchvision.transforms import transforms as T
from transformers import CLIPProcessor, CLIPVisionModel

try:
    import dt_adapters.envs.rlbench_env
except:
    pass

# import dt_adapters.mw_utils as mw_utils
from dt_adapters.envs.make_env import env_constructor
import dt_adapters.general_utils as general_utils
import dt_adapters.data.utils as data_utils
from dt_adapters.envs.video_recorder import VideoRecorder


def mp_rollout(config, model, **kwargs):
    rollout_kwargs = general_utils.AttrDict(
        task=config.data.eval_task,
        config=config,
        model=model,
        # state_mean=self.dataset.state_mean,
        # state_std=self.dataset.state_std,
        device="cuda",
        **kwargs,
    )

    eval_rollouts = []
    if config.num_processes > 0:
        # run multiple threads at the same time
        p = mp.Pool(processes=config.num_processes)

        # fetch the results of the rollout
        results = [
            p.apply_async(rollout, (), rollout_kwargs)
            for i in range(config.num_eval_rollouts)
        ]
        eval_rollouts = [p.get() for p in results]
        p.close()
        p.join()
    else:
        for i in range(config.num_eval_rollouts):
            eval_rollouts.append(rollout(**rollout_kwargs))

    return eval_rollouts


def rollout(
    task,
    config,
    model,
    # state_mean,
    # state_std,
    device=None,
    use_means=False,
    attend_to_rtg=False,
    log_eval_videos=False,
    **kwargs,
):
    observation_mode = config.data.observation_mode
    if config.data.env_name == "metaworld":
        env = env_constructor(env_name=task, config=config, device=device)
    elif config.data.env_name == "rlbench":
        env = gym.make(
            f"{task}-image-v0",
            config=config.data,
            render_mode="rgb_array",
        )

    if log_eval_videos:
        recorder = VideoRecorder(env, path="video.mp4")
    else:
        recorder = None

    model.reset()

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)
    # state_mean = None
    # state_std = None

    state_dim = model.state_dim
    act_dim = model.act_dim

    with torch.no_grad():
        start = time.time()

        # ============= PROCESS FIRST OBS =============
        last_obs = env.reset()

        if log_eval_videos:
            recorder.capture_frame()

        # create initial conditioning information
        # these tensors will store the context history for inputting to the model
        states = (
            torch.from_numpy(last_obs)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )

        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_length = 0
        traj_success = False

        while episode_length < (config.data.max_ep_len or np.inf):
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )

            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                states=states,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=None,
                obj_ids=None,
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
            )

            actions[-1] = action
            action = action.detach().cpu().numpy()

            obs, reward, terminate, info = env.step(action)

            if log_eval_videos:
                recorder.capture_frame()

            if config.data.env_name == "metaworld":
                terminate |= bool(info["success"])

            if terminate:
                traj_success = True

            episode_length += 1
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * episode_length,
                ],
                dim=1,
            )

            last_obs = obs
            cur_state = (
                torch.from_numpy(last_obs).to(device=device).reshape(1, state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = reward
            if terminate:
                break

        rollout_time = time.time() - start

    out = dict(
        states=general_utils.to_numpy(states),
        actions=general_utils.to_numpy(actions),
        rewards=general_utils.to_numpy(rewards),
        rollout_time=rollout_time,
        frames=np.array(recorder.recorded_frames) if recorder is not None else None,
        traj_success=traj_success,
    )

    # clean up
    # env.close()
    del env
    gc.collect()
    return out


# import multiprocessing as mp
# import time as timer


# def sample_paths(
#     task,
#     config,
#     num_traj,
#     policy,
#     eval_mode=False,
#     horizon=1e6,
#     base_seed=None,
#     num_cpu=1,
#     max_process_time=300,
#     max_timeouts=4,
#     suppress_print=False,
#     env_kwargs=None,
# ):

#     num_cpu = 1 if num_cpu is None else num_cpu
#     num_cpu = mp.cpu_count() if num_cpu == "max" else num_cpu
#     assert type(num_cpu) == int

#     if num_cpu == 1:
#         input_dict = dict(
#             task=task,
#             config=config,
#             num_traj=num_traj,
#             model=policy,
#             device="cuda"
#             # eval_mode=eval_mode,
#             # horizon=horizon,
#             # base_seed=base_seed,
#             # env_kwargs=env_kwargs,
#         )
#         # dont invoke multiprocessing if not necessary
#         return rollout(**input_dict)

#     # do multiprocessing otherwise
#     paths_per_cpu = int(np.ceil(num_traj / num_cpu))
#     input_dict_list = []
#     for i in range(num_cpu):
#         input_dict = dict(
#             task=task,
#             config=config,
#             num_traj=paths_per_cpu,
#             model=policy,
#             device="cuda"
#             # eval_mode=eval_mode,
#             # horizon=horizon,
#             # base_seed=base_seed + i * paths_per_cpu,
#             # env_kwargs=env_kwargs,
#         )
#         input_dict_list.append(input_dict)
#     if suppress_print is False:
#         start_time = timer.time()
#         print("####### Gathering Samples #######")

#     results = _try_multiprocess(
#         rollout, input_dict_list, num_cpu, max_process_time, max_timeouts
#     )
#     if suppress_print is False:
#         print(
#             "======= Samples Gathered  ======= | >>>> Time taken = %f "
#             % (timer.time() - start_time)
#         )

#     return results


# def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
#     # Base case
#     print(f"multprocessing rollouts, num_cpu: {num_cpu}")
#     if max_timeouts == 0:
#         return None

#     pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
#     parallel_runs = [
#         pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list
#     ]
#     try:
#         results = [p.get(timeout=max_process_time) for p in parallel_runs]
#     except Exception as e:
#         print(str(e))
#         print("Timeout Error raised... Trying again")
#         pool.close()
#         pool.terminate()
#         pool.join()
#         return _try_multiprocess(
#             func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1
#         )

#     pool.close()
#     pool.terminate()
#     pool.join()
#     return results
