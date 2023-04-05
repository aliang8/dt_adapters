from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from collections import namedtuple
from dt_adapters.envs.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from dt_adapters.envs.gym_env import GymEnv
from typing import Optional, Tuple, Dict, Union, List


def env_constructor(
    domain: str,
    env_name: str,
    image_keys: List[str] = [],
    vision_backbone: str = None,
    image_width: int = 256,
    image_height: int = 256,
    proprio: int = 0,
    device: str = "cpu",
) -> GymEnv:
    """
    Creates environment class and wrappers for using image observations.
    """

    # need to do some special environment config for the metaworld environments
    if domain == "metaworld":
        env = ALL_V2_ENVIRONMENTS[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        env._partially_observable = False
    else:
        print(f"{domain} not supported")

    if len(image_keys) > 0:
        # wrap in pixel observation wrapper
        env = MuJoCoPixelObs(
            env,
            width=image_width,
            height=image_height,
            camera_names=image_keys,
            device_id=0,
        )

        # wrapper which encodes state in pretrained model
        env = StateEmbedding(
            env,
            vision_backbone=vision_backbone,
            device=device,
            proprio=proprio,
        )
        env = GymEnv(env)

    return env
