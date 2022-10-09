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

import dt_adapters.envs.rlbench_env
import dt_adapters.general_utils as general_utils
from dt_adapters.data.process_rlbench_data import extract_image_feats


def rollout(
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
    results_queue=None,
):
    env = gym.make(
        f"{config.task}-vision-v0", config=config.data, render_mode="rgb_array"
    )

    model.reset()

    agent_infos = []
    episode_infos = []
    frames = dd(list)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    state_dim = model.state_dim
    act_dim = model.act_dim

    with torch.no_grad():
        start = time.time()
        print(f"starting rollout..., num_steps: {config.data.max_ep_len}")

        # not sure why i can't pickle this stuff
        img_preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        depth_img_preprocessor = T.Compose(
            [
                T.Lambda(
                    lambda images: torch.stack(
                        [T.ToTensor()(image) for image in images]
                    )
                ),
                T.Resize([64]),
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.Lambda(lambda images: images.numpy()),
            ]
        )

        # ============= PROCESS FIRST OBS =============
        last_obs = env.reset()
        ll_state = last_obs["state"]
        img_obs = last_obs
        del img_obs["state"]

        # add batch dimension
        for k, _ in img_obs.items():
            img_obs[k] = img_obs[k][np.newaxis]

        img_feats = extract_image_feats(
            img_obs,
            img_preprocessor,
            img_encoder,
            depth_img_preprocessor,
            depth_img_encoder,
        )

        [frames[k].append(v) for k, v in img_obs.items()]

        if log_eval_videos:
            frames["render"].append(env.render(mode="rgb_array"))

        last_obs, last_img_feats = ll_state, img_feats

        states = (
            torch.from_numpy(last_obs)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )
        img_feats = (
            torch.from_numpy(last_img_feats)
            .reshape(1, -1)
            .to(device, dtype=torch.float32)
        )

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

        while episode_length < (config.data.max_ep_len or np.inf):
            # print(episode_length)
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action, _, agent_info = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=None,
                obj_ids=None,
                img_feats=img_feats.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
                use_rtg_mask=use_rtg_mask,
                sample_return_dist=config.model.predict_return_dist,
            )

            actions[-1] = action
            action = action.detach().cpu().numpy()

            obs, reward, terminate, info = env.step(action)

            # ============= PROCESS CURRENT OBS =============
            last_obs = ll_state = obs["state"]
            img_obs = obs
            del img_obs["state"]

            # add batch dimension
            for k, _ in img_obs.items():
                img_obs[k] = img_obs[k][np.newaxis]

            last_img_feats = extract_image_feats(
                img_obs,
                img_preprocessor,
                img_encoder,
                depth_img_preprocessor,
                depth_img_encoder,
            )

            [frames[k].append(v) for k, v in img_obs.items()]

            if log_eval_videos:
                frames["render"].append(env.render(mode="rgb_array"))

            agent_infos.append(agent_info)

            episode_length += 1
            if terminate:
                break

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * episode_length,
                ],
                dim=1,
            )
            cur_state = (
                torch.from_numpy(last_obs).to(device=device).reshape(1, state_dim)
            )
            cur_img_feats = (
                torch.from_numpy(last_img_feats).to(device=device).reshape(1, -1)
            )
            states = torch.cat([states, cur_state], dim=0)
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

        for k, v in frames.items():
            if len(v[0].shape) == 3:
                frames[k] = np.stack(frames[k])
            elif len(v[0].shape) == 4:
                frames[k] = np.concatenate(frames[k])

        rollout_time = time.time() - start

    out = dict(
        episode_infos=episode_infos,
        states=general_utils.to_numpy(states),
        rewards=general_utils.to_numpy(rewards),
        frames=frames,
        rollout_time=rollout_time,
    )
    env.close()

    if results_queue:
        results_queue.put(out)
    return out
