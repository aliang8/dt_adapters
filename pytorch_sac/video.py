import imageio
import os
import numpy as np
import sys

import utils
import wandb


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            # frame = env.render(mode='rgb_array',
            #                    height=self.height,
            #                    width=self.width,
            #                    camera_id=self.camera_id)
            frame = env.sim.render(
                height=self.height, width=self.width, camera_name="corner"
            )

            self.frames.append(frame)

    def save(self, file_name, step, logger):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print(path)
            imageio.mimsave(path, self.frames, fps=self.fps)

            if logger is not None:
                logger.log_video(
                    "rollout_video", np.array(self.frames).transpose(0, 3, 1, 2), step
                )
