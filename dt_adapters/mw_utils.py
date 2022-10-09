import numpy as np
from PIL import Image
from dt_adapters.general_utils import split
from metaworld.policies import *
from dt_adapters.mw_constants import *
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

ENVS_AND_SCRIPTED_POLICIES = [
    # name, policy, action noise pct, success rate
    ["assembly-v2", SawyerAssemblyV2Policy(), 0.1, 0.70],
    ["basketball-v2", SawyerBasketballV2Policy(), 0.1, 0.96],
    ["bin-picking-v2", SawyerBinPickingV2Policy(), 0.1, 0.96],
    ["box-close-v2", SawyerBoxCloseV2Policy(), 0.1, 0.82],
    ["button-press-topdown-v2", SawyerButtonPressTopdownV2Policy(), 0.1, 0.93],
    ["button-press-topdown-wall-v2", SawyerButtonPressTopdownWallV2Policy(), 0.1, 0.95],
    ["button-press-v2", SawyerButtonPressV2Policy(), 0.1, 0.98],
    ["button-press-wall-v2", SawyerButtonPressWallV2Policy(), 0.1, 0.92],
    ["coffee-button-v2", SawyerCoffeeButtonV2Policy(), 0.1, 0.99],
    ["coffee-pull-v2", SawyerCoffeePullV2Policy(), 0.1, 0.82],
    ["coffee-push-v2", SawyerCoffeePushV2Policy(), 0.1, 0.88],
    ["dial-turn-v2", SawyerDialTurnV2Policy(), 0.1, 0.84],
    ["disassemble-v2", SawyerDisassembleV2Policy(), 0.1, 0.88],
    ["door-close-v2", SawyerDoorCloseV2Policy(), 0.1, 0.97],
    ["door-lock-v2", SawyerDoorLockV2Policy(), 0.1, 0.96],
    ["door-open-v2", SawyerDoorOpenV2Policy(), 0.1, 0.92],
    ["door-unlock-v2", SawyerDoorUnlockV2Policy(), 0.1, 0.97],
    ["drawer-close-v2", SawyerDrawerCloseV2Policy(), 0.1, 0.99],
    ["drawer-open-v2", SawyerDrawerOpenV2Policy(), 0.1, 0.97],
    ["faucet-close-v2", SawyerFaucetCloseV2Policy(), 0.1, 1.0],
    ["faucet-open-v2", SawyerFaucetOpenV2Policy(), 0.1, 0.99],
    ["hammer-v2", SawyerHammerV2Policy(), 0.1, 0.96],
    ["hand-insert-v2", SawyerHandInsertV2Policy(), 0.1, 0.86],
    ["handle-press-side-v2", SawyerHandlePressSideV2Policy(), 0.1, 0.98],
    ["handle-press-v2", SawyerHandlePressV2Policy(), 0.1, 1.0],
    ["handle-pull-v2", SawyerHandlePullV2Policy(), 0.1, 0.99],
    ["handle-pull-side-v2", SawyerHandlePullSideV2Policy(), 0.1, 0.71],
    ["peg-insert-side-v2", SawyerPegInsertionSideV2Policy(), 0.1, 0.87],
    ["lever-pull-v2", SawyerLeverPullV2Policy(), 0.1, 0.90],
    ["peg-unplug-side-v2", SawyerPegUnplugSideV2Policy(), 0.1, 0.80],
    ["pick-out-of-hole-v2", SawyerPickOutOfHoleV2Policy(), 0.1, 0.89],
    ["pick-place-v2", SawyerPickPlaceV2Policy(), 0.1, 0.83],
    ["pick-place-wall-v2", SawyerPickPlaceWallV2Policy(), 0.1, 0.83],
    ["plate-slide-back-side-v2", SawyerPlateSlideBackSideV2Policy(), 0.1, 0.95],
    ["plate-slide-back-v2", SawyerPlateSlideBackV2Policy(), 0.1, 0.94],
    ["plate-slide-side-v2", SawyerPlateSlideSideV2Policy(), 0.1, 0.78],
    ["plate-slide-v2", SawyerPlateSlideV2Policy(), 0.1, 0.97],
    ["reach-v2", SawyerReachV2Policy(), 0.1, 0.98],
    ["reach-wall-v2", SawyerReachWallV2Policy(), 0.1, 0.96],
    ["push-back-v2", SawyerPushBackV2Policy(), 0.0, 0.91],
    ["push-v2", SawyerPushV2Policy(), 0.1, 0.88],
    ["push-wall-v2", SawyerPushWallV2Policy(), 0.1, 0.82],
    ["shelf-place-v2", SawyerShelfPlaceV2Policy(), 0.1, 0.89],
    ["soccer-v2", SawyerSoccerV2Policy(), 0.1, 0.81],
    ["stick-pull-v2", SawyerStickPullV2Policy(), 0.1, 0.81],
    ["stick-push-v2", SawyerStickPushV2Policy(), 0.1, 0.95],
    ["sweep-into-v2", SawyerSweepIntoV2Policy(), 0.1, 0.86],
    ["sweep-v2", SawyerSweepV2Policy(), 0.0, 0.99],
    ["window-close-v2", SawyerWindowCloseV2Policy(), 0.1, 0.95],
    ["window-open-v2", SawyerWindowOpenV2Policy(), 0.1, 0.93],
]


def initialize_env(
    task, obj_randomization=False, hide_goal=False, observation_mode="state"
):
    e = ALL_V2_ENVIRONMENTS[task]()
    e._partially_observable = hide_goal
    e._freeze_rand_vec = False
    e._set_task_called = True
    e._observation_mode = observation_mode

    if not obj_randomization:
        e.reset()
        e._freeze_rand_vec = True
    return e


def get_object_indices(env):
    # get object indices
    objects_in_env = ENV_TO_OBJECTS[
        env.replace("-goal-observable", "").replace("-", "_")
    ]
    object_indices = [0, 0]
    for i, obj in enumerate(objects_in_env):
        object_indices[i] = OBJECTS.index(obj)
    return np.array(object_indices)


def create_video_grid(videos, height=64, width=64, max_columns=5):
    # wandb needs videos to be in BxCxHxW
    # assert len(videos) % max_columns == 0
    if len(videos) % max_columns != 0:
        # need to pad with some black videos
        extra_videos_needed = ((len(videos) // max_columns) + 1) * max_columns - len(
            videos
        )
        for i in range(extra_videos_needed):
            videos.append(np.zeros_like(videos[0]))

    assert len(videos) % max_columns == 0

    max_seq_length = max([video.shape[0] for video in videos])

    # first resize videos and pad them to max length
    for i, video in enumerate(videos):
        all_frames = []
        for frame in video:
            if frame.shape[0] == 1:
                frame = (
                    frame.reshape((frame.shape[1], frame.shape[2], 1)).repeat(
                        3, axis=-1
                    )
                    * 256
                ).astype(np.uint8)
            frame = Image.fromarray(frame)
            frame = np.array(frame)
            # frame = np.array(frame.resize((height, width)))
            all_frames.append(frame)
        all_frames = np.array(all_frames).transpose(0, 3, 1, 2)
        video = all_frames

        if video.shape[0] < max_seq_length:
            padded_video = np.zeros((max_seq_length, *all_frames.shape[1:]))
            padded_video[: video.shape[0]] = video
            videos[i] = padded_video
        else:
            videos[i] = video

    max_columns = 5
    num_rows = int(len(videos) / max_columns)
    chunks = list(split(videos, num_rows))

    rows = []
    for chunk in chunks:
        # stick the videos into a grid and concatenate the videos on width
        row = np.concatenate(chunk, axis=-1)
        rows.append(row)

    videos = np.concatenate(rows, axis=-2)  # concat over height
    return videos
