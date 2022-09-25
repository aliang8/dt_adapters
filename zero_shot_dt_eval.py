import os
import glob
import wandb
import torch
import hydra
import numpy as np
import time
from utils import split
import collections
from collections import defaultdict as dd

from mw_utils import ENVS_AND_SCRIPTED_POLICIES, initialize_env, create_video_grid
from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list
from models.decision_transformer import DecisionTransformerSeparateState
import multiprocessing as mp
from mw_dataset import OBJECTS_TO_ENV, ENV_TO_OBJECTS, OBJECTS
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from PIL import Image
from mw_dataset import MWDemoDataset
from utils import create_exp_prefix, KEYS_TO_USE


def rollout(
    env_name,
    env,
    agent,
    state_mean,
    state_std,
    max_episode_length=np.inf,
    animated=False,
    device="cpu",
    target_return=None,
    pause_per_frame=None,
    deterministic=False,
):
    """Sample a single episode of the agent in the environment."""
    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()

    state_dim = agent.state_dim
    act_dim = agent.act_dim
    if animated:
        env.visualize()

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    states = (
        torch.from_numpy(last_obs)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(
        target_return, device=device, dtype=torch.float32
    ).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    objects_in_env = ENV_TO_OBJECTS[
        env_name.replace("-goal-observable", "").replace("-", "_")
    ]
    object_indices = [0, 0]
    for i, obj in enumerate(objects_in_env):
        object_indices[i] = OBJECTS.index(obj)
    object_indices = torch.tensor(object_indices).long().to(device)

    episode_length = 0

    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action, agent_info = agent.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions.to(dtype=torch.float32),
            returns_to_go=target_return,
            obj_ids=object_indices.long().reshape(1, object_indices.shape[0]),
            timesteps=timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        import ipdb

        ipdb.set_trace()
        action = action.detach().cpu().numpy()

        # if deterministic and "mean" in agent_info:
        #     action = agent_info["mean"]

        es = env.step(action)

        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)

        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long)
                * (episode_length + 1),
            ],
            dim=1,
        )
        cur_state = torch.from_numpy(last_obs).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)

        # pred_return = target_return[0, -1] - (reward / scale)
        pred_return = target_return[0, -1] - es.reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        rewards[-1] = es.reward

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )


def zero_shot_eval(cfg, runs, results_queue, wandb_run, state_mean, state_std):
    for (env_name, ckpt_file) in runs:
        env_name = env_name.replace("_", "-")
        assert "v2" in env_name

        # load model
        print(f"loading model from {ckpt_file}")
        state_dict = torch.load(ckpt_file)
        model_cfg = state_dict["config"]

        policy = DecisionTransformerSeparateState(**model_cfg.model)
        del state_dict["config"]
        del state_dict["epoch"]
        policy.load_state_dict(state_dict, strict=True)
        policy.to(cfg.device)
        policy = policy.eval()

        # create environment
        env = initialize_env(env_name)
        max_path_length = env.max_path_length
        env = GymEnv(env, max_episode_length=max_path_length)

        with torch.no_grad():
            # collect rollout
            path = rollout(
                env_name,
                env,
                policy,
                state_mean,
                state_std,
                max_episode_length=np.inf,
                animated=True,
                target_return=cfg.target_return,
                pause_per_frame=None,
                device=cfg.device,
                deterministic=False,
            )

            if results_queue is not None:
                results_queue.put((env_name, path))


def handle_output(cfg, results_queue, wandb_run):
    stats = dd(lambda: dd(int))
    videos = dd(list)

    while True:
        out = results_queue.get()
        if out is not None:
            env_name, path = out
            videos[env_name].append(path["env_infos"]["frames"])

            if len(videos[env_name]) == cfg.num_rollout_per_env and cfg.log_to_wandb:
                print(f"done with {env_name}, saving videos")
                start = time.time()
                video_array = create_video_grid(videos[env_name], height=128, width=128)
                wandb_run.log(
                    {
                        f"{env_name}/rollout_videos": wandb.Video(
                            video_array, fps=10, format="gif"
                        )
                    }
                )
                print(f"took {time.time() - start}")

            success = np.any(path["env_infos"]["success"])
            if success:
                stats[env_name]["success_rate"] += 1
        else:
            break

    if cfg.log_to_wandb:
        data = [[env, stats["success_rate"]] for env, stats in stats.items()]
        table = wandb.Table(data=data, columns=["env_name", "sr"])
        wandb_run.log(
            {
                "success_rates": wandb.plot.bar(
                    table, "env_name", "sr", title="Success Rates"
                )
            }
        )


@hydra.main(config_path="configs", config_name="eval")
def main(config):
    runs = []
    envs_to_evaluate = []
    if config.filter_envs_by_obj:
        envs_to_evaluate = OBJECTS_TO_ENV[config.filter_envs_by_obj]
    else:
        envs_to_evaluate = ["reach-v2"]

    model_ckpts = []
    # ckpt_dir = os.path.join(config.output_dir, config.exp_name)
    # exps = glob.glob(f"{ckpt_dir}/*")
    # for exp in exps:
    #     ckpt_file = sorted(glob.glob(f"{exp}/models/*"))[-1]
    #     model_ckpts.append(ckpt_file)

    # print(f"found {len(model_ckpts)} model ckpts")
    ckpt_file = sorted(glob.glob(f"{config.model_ckpt_dir}/models/*"))[-1]
    model_ckpts = [ckpt_file]

    if config.log_to_wandb:
        wandb_run = wandb.init(
            name=model_ckpts[0].replace(config.output_dir, ""),
            group="zero_shot_eval",
            project="dt-adapters",
            config={},
            entity="glamor",
        )
    else:
        wandb_run = None

    for e in range(config.num_rollout_per_env):
        for model_ckpt in model_ckpts:
            for env in envs_to_evaluate:
                runs.append((env, model_ckpt))

    print(f"number of runs: {len(runs)}")

    dataset = MWDemoDataset(config.data)
    state_mean, state_std = dataset.state_mean, dataset.state_std

    if config.num_processes > 0:
        torch.multiprocessing.set_start_method("spawn")
        results_queue = mp.Queue()

        proc = mp.Process(
            target=handle_output,
            args=(config, results_queue, wandb_run),
        )

        processes = []
        proc.start()

        runs = list(split(runs, config.num_processes))

        for rank in range(config.num_processes):
            p = mp.Process(
                target=zero_shot_eval,
                args=(
                    config,
                    runs[rank],
                    results_queue,
                    wandb_run,
                    state_mean,
                    state_std,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results_queue.put(None)
        proc.join()
    else:
        zero_shot_eval(config, [runs[0]], None, None, state_mean, state_std)


if __name__ == "__main__":
    main()
