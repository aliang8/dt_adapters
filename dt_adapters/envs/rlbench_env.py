from typing import Union, Dict, Tuple

import gym
import numpy as np
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config,
        task_class,
        observation_mode="state",
        render_mode: Union[None, str] = None,
        **kwargs
    ):
        self.config = config
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if  "state" in observation_mode:
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif "image" in observation_mode:
            obs_config.set_all(True)
        else:
            raise ValueError("Unrecognised observation_mode: %s." % observation_mode)

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self.env = Environment(action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.env.action_shape)

        if observation_mode == "state":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape
            )
        elif observation_mode == "image":
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape
                    ),
                    "left_shoulder_rgb": spaces.Box(
                        low=0, high=1, shape=obs.left_shoulder_rgb.shape
                    ),
                    "left_shoulder_depth": spaces.Box(
                        low=0, high=1, shape=obs.left_shoulder_depth.shape
                    ),
                    "right_shoulder_rgb": spaces.Box(
                        low=0, high=1, shape=obs.right_shoulder_rgb.shape
                    ),
                    "right_shoulder_depth": spaces.Box(
                        low=0, high=1, shape=obs.right_shoulder_depth.shape
                    ),
                    "wrist_rgb": spaces.Box(low=0, high=1, shape=obs.wrist_rgb.shape),
                    "wrist_depth": spaces.Box(low=0, high=1, shape=obs.wrist_depth.shape),
                    "front_rgb": spaces.Box(low=0, high=1, shape=obs.front_rgb.shape),
                    "front_depth": spaces.Box(low=0, high=1, shape=obs.front_depth.shape),
                    "overhead_rgb": spaces.Box(
                        low=0, high=1, shape=obs.overhead_rgb.shape
                    ),
                    "overhead_depth": spaces.Box(
                        low=0, high=1, shape=obs.overhead_depth.shape
                    ),
                }
            )

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs):
        ll_state_info = [
            np.array(getattr(obs, k)).reshape(-1) for k in self.config.ll_state_keys
        ]
        ll_state = np.concatenate(ll_state_info)

        if self._observation_mode == "state":
            # return obs.get_low_dim_data()
            return {"state": ll_state}
        elif self._observation_mode == "image":
            image_state = {k: getattr(obs, k) for k in self.config.image_keys}
            return {"state": ll_state, **image_state}
            # return {
            #     "state": obs.get_low_dim_data(),
            #     "left_shoulder_rgb": obs.left_shoulder_rgb,
            #     "right_shoulder_rgb": obs.right_shoulder_rgb,
            #     "wrist_rgb": obs.wrist_rgb,
            #     "front_rgb": obs.front_rgb,
            # }

    def render(self, mode="human"):
        if mode != self._render_mode:
            raise ValueError(
                "The render mode must match the render mode selected in the "
                'constructor. \nI.e. if you want "human" render mode, then '
                "create the env by calling: "
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                "You passed in mode %s, but expected %s." % (mode, self._render_mode)
            )
        if mode == "rgb_array":
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
            return frame

    def reset(self):
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        info = {}
        info["success"] = terminate
        return self._extract_obs(obs), reward, terminate, info

    def close(self):
        self.env.shutdown()
