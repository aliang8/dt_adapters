from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from collections import namedtuple
from dt_adapters.envs.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from dt_adapters.envs.gym_env import GymEnv


def env_constructor(
    env_name,
    config,
    device="cuda",
    # image_width=256,
    # image_height=256,
    # camera_name=None,
    # embedding_name="resnet50",
    # pixel_based=True,
    # render_gpu_id=0,
    # load_path="",
    # proprio=False,
    # lang_cond=False,
    # gc=False,
):

    ## Need to do some special environment config for the metaworld environments
    # if "v2" in env_name:
    e = ALL_V2_ENVIRONMENTS[env_name]()
    e._freeze_rand_vec = False
    e._set_task_called = True
    e._partially_observable = False
    # e.spec = namedtuple("spec", ["id", "max_episode_steps"])
    # e.spec.id = env_name
    # e.spec.max_episode_steps = 500

    if "image" in config.data.observation_mode:
        ## Wrap in pixel observation wrapper
        e = MuJoCoPixelObs(
            e,
            width=config.data.image_width,
            height=config.data.image_height,
            camera_names=config.data.image_keys,
            device_id=0,
        )
        ## Wrapper which encodes state in pretrained model
        e = StateEmbedding(
            e,
            vision_backbone=config.data.vision_backbone,
            device=device,
            proprio=config.data.proprio,
        )
        e = GymEnv(e)

    return e
