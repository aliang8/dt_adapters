from dt_adapters.envs.gym_env import GymEnv
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from collections import namedtuple
from typing import Optional, Tuple, Dict, Union, List


def env_constructor(
    domain: str,
    task_name: str,
):
    """
    Creates environment class and wrappers for using image observations.
    """

    # need to do some special environment config for the metaworld environments
    if domain == "metaworld":
        env = ALL_V2_ENVIRONMENTS[task_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        env._partially_observable = False
    else:
        print(f"{domain} not supported")

    # wrap the environment with a gym environment
    env = GymEnv(env)

    return env
