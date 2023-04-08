import torch
import numpy as np
import dt_adapters.utils.utils as utils
from dt_adapters.models.transformer_policy import TrajectoryModel
from dt_adapters.utils.utils import AttrDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union


def run_single_rollout(
    env,
    domain: str,
    model: TrajectoryModel,
    device: str,
    max_episode_length: Optional[int] = None,
    save_frames: bool = False,
    image_dim: int = 64,
    camera_names: List[str] = [],
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    **kwargs,
) -> AttrDict:
    """
    Run a single episode of rollout
    """
    episode_length = 0
    success = False
    last_obs = env.reset()

    # create initial conditioning information
    # these tensors will store the context history for inputting to the model
    states = (
        torch.from_numpy(last_obs)
        .reshape(1, env.observation_dim)
        .to(device=device, dtype=torch.float32)
    )

    actions = torch.zeros((0, env.action_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor([0], device=device, dtype=torch.long)

    if save_frames:
        frames = {
            f"{camera_name}": [env.render(image_dim=image_dim, camera_name=camera_name)]
            for camera_name in camera_names
        }

    while episode_length < (max_episode_length or np.inf):
        # add placeholder for next action
        action_pad = torch.zeros((1, env.action_dim), device=device)
        actions = torch.cat([actions, action_pad], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states - state_mean) / state_std,
            actions=actions,
            timesteps=timesteps,
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        obs, reward, terminate, info = env.step(action)

        if domain == "metaworld":
            terminate |= bool(info["success"])

        if terminate:
            success = True

        episode_length += 1

        timestep_pad = torch.ones((1), device=device, dtype=torch.long) * episode_length
        timesteps = torch.cat([timesteps, timestep_pad], dim=0)

        last_obs = obs

        if save_frames:
            last_frames = {
                f"{camera_name}": env.render(
                    image_dim=image_dim, camera_name=camera_name
                )
                for camera_name in camera_names
            }

            for k, v in last_frames.items():
                frames[k].append(v)

        cur_state = (
            torch.from_numpy(last_obs).to(device=device).reshape(1, env.observation_dim)
        )
        states = torch.cat([states, cur_state], dim=0)

        rewards[-1] = reward

        if terminate:
            break

    if save_frames:
        for k, v in frames.items():
            frames[k] = np.array(v)

    trajectory = AttrDict(
        states=utils.to_numpy(states),
        actions=utils.to_numpy(actions),
        rewards=utils.to_numpy(rewards),
        images=frames,
        success=success,
    )
    return trajectory
