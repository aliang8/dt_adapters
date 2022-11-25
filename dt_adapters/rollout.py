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
from collections import defaultdict as dd

import gym

from torchvision.transforms import transforms as T
from transformers import CLIPProcessor, CLIPVisionModel

try:
    import dt_adapters.envs.rlbench_env
except:
    pass

import dt_adapters.mw_utils as mw_utils
import dt_adapters.general_utils as general_utils
import dt_adapters.data.utils as data_utils


def rollout(
    task,
    config,
    model,
    img_encoder,
    depth_img_encoder,
    state_mean,
    state_std,
    device=None,
    use_means=False,
    attend_to_rtg=False,
    log_eval_videos=False,
):
    observation_mode = config.data.observation_mode
    if config.data.env_name == "metaworld":
        env = mw_utils.initialize_env(
            task=task,
            obj_randomization=config.data.obj_randomization,
            hide_goal=False,
            observation_mode=observation_mode,
        )
    elif config.data.env_name == "rlbench":
        env = gym.make(
            f"{task}-image-v0",
            config=config.data,
            render_mode="rgb_array",
        )

    model.reset()

    agent_infos = []
    episode_infos = []
    frames = dd(list)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    state_dim = model.state_dim
    act_dim = model.act_dim

    if "image" in observation_mode:
        img_preprocessor, depth_img_preprocessor = data_utils.get_preprocessor(
            config.model.state_encoder.vision_backbone,
        )

    with torch.no_grad():
        start = time.time()

        # ============= PROCESS FIRST OBS =============
        last_obs = env.reset()

        if "state" in observation_mode:
            ll_state = last_obs["state"]
        else:
            ll_state = None

        if "image" in observation_mode:
            img_obs = last_obs
            if "state" in img_obs:
                del img_obs["state"]

            # add batch dimension
            for k, _ in img_obs.items():
                img_obs[k] = img_obs[k][np.newaxis]

            img_feats = data_utils.extract_image_feats(
                img_obs,
                img_preprocessor,
                img_encoder,
                depth_img_preprocessor,
                depth_img_encoder,
                vision_backbone=config.data.vision_backbone,
            )

            [frames[k].append(v) for k, v in img_obs.items()]
            last_img_feats = img_feats

        last_obs = ll_state

        if log_eval_videos:
            if config.data.env_name == "metaworld":
                frame = env.sim.render(height=300, width=300, camera_name="corner")
            else:
                frame = env.render(mode="rgb_array")

            frames["render"].append(frame)

        # create initial conditioning information
        # these tensors will store the context history for inputting to the model
        if "state" in observation_mode:
            states = (
                torch.from_numpy(last_obs)
                .reshape(1, state_dim)
                .to(device=device, dtype=torch.float32)
            )
        else:
            states = None

        if "image" in observation_mode:
            img_feats = last_img_feats.reshape(1, -1).to(device, dtype=torch.float32)
        else:
            img_feats = None

        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        use_rtg_mask = torch.tensor([attend_to_rtg]).reshape(1, 1).to(device)

        # target_return = torch.tensor(
        #     config.target_return / config.scale,
        #     device=device,
        #     dtype=torch.float32,
        # ).reshape(1, 1)

        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        episode_length = 0
        traj_success = False

        while episode_length < (config.data.max_ep_len or np.inf):
            # print(episode_length)
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )

            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            if "state" in observation_mode:
                input_states = (states.to(dtype=torch.float32) - state_mean) / state_std
            else:
                input_states = None

            action, _, agent_info = model.get_action(
                states=input_states,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=None,
                obj_ids=None,
                img_feats=img_feats,
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
                use_rtg_mask=use_rtg_mask,
                sample_return_dist=config.model.predict_return_dist,
            )

            actions[-1] = action
            action = action.detach().cpu().numpy()

            obs, reward, terminate, info = env.step(action)

            if config.data.env_name == "metaworld":
                terminate |= bool(info["success"])

            if terminate:
                traj_success = True

            # ============= PROCESS CURRENT OBS =============
            if "state" in observation_mode:
                ll_state = obs["state"]
            else:
                ll_state = None

            if "image" in observation_mode:
                img_obs = obs
                if "state" in img_obs:
                    del img_obs["state"]

                # add batch dimension
                for k, _ in img_obs.items():
                    img_obs[k] = img_obs[k][np.newaxis]

                last_img_feats = data_utils.extract_image_feats(
                    img_obs,
                    img_preprocessor,
                    img_encoder,
                    depth_img_preprocessor,
                    depth_img_encoder,
                    vision_backbone=config.data.vision_backbone,
                )

                [frames[k].append(v) for k, v in img_obs.items()]

            last_obs = ll_state

            if log_eval_videos:
                if config.data.env_name == "metaworld":
                    frame = env.sim.render(height=300, width=300, camera_name="corner")
                else:
                    frame = env.render(mode="rgb_array")

                frames["render"].append(frame)

            agent_infos.append(agent_info)

            episode_length += 1
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * episode_length,
                ],
                dim=1,
            )

            if "state" in observation_mode:
                cur_state = (
                    torch.from_numpy(last_obs).to(device=device).reshape(1, state_dim)
                )
                states = torch.cat([states, cur_state], dim=0)

            if "image" in observation_mode:
                cur_img_feats = last_img_feats.to(device=device).reshape(1, -1)
                img_feats = torch.cat([img_feats, cur_img_feats], dim=0)

            # if config.model.predict_return_dist:
            #     # use the model's prediction of the return to go
            #     # follow this paper: https://openreview.net/forum?id=fwJWhOxuzV9
            #     target_return = torch.cat(
            #         [target_return, return_target.reshape(1, 1)], dim=1
            #     )
            # else:
            #     pred_return = target_return[0, -1] - (es.reward / config.scale)
            #     target_return = torch.cat(
            #         [target_return, pred_return.reshape(1, 1)], dim=1
            #     )

            rewards[-1] = reward
            if terminate:
                break

        for k, v in frames.items():
            if len(v[0].shape) == 3:
                frames[k] = np.stack(frames[k])
            elif len(v[0].shape) == 4:
                frames[k] = np.concatenate(frames[k])

        rollout_time = time.time() - start

    if states is not None:
        states = general_utils.to_numpy(states)

    out = dict(
        episode_infos=episode_infos,
        states=states,
        actions=general_utils.to_numpy(actions),
        rewards=general_utils.to_numpy(rewards),
        frames=frames,
        rollout_time=rollout_time,
        traj_success=traj_success,
    )
    env.close()
    return out
