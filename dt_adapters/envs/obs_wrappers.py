import pickle
import gym
from gym.spaces.box import Box

import numpy as np
import omegaconf

import torch
import torch.nn as nn
import numpy as np

from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T

from PIL import Image
from pathlib import Path
from torchvision.utils import save_image


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _get_embedding(vision_backbone="resnet34", *args, **kwargs):
    if vision_backbone == "resnet34":
        model = models.resnet34(pretrained=True, progress=False)
        embedding_dim = 512
    elif vision_backbone == "resnet18":
        model = models.resnet18(pretrained=True, progress=False)
        embedding_dim = 512
    elif vision_backbone == "resnet50":
        model = models.resnet50(pretrained=True, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim


class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        vision_backbone (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """

    def __init__(
        self,
        env,
        vision_backbone=None,
        device="cuda",
        proprio=0,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.vision_backbone = vision_backbone
        self.start_finetune = False
        self.device = device

        if vision_backbone == "clip":
            import clip

            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif "resnet" in vision_backbone:
            embedding, embedding_dim = _get_embedding(vision_backbone=vision_backbone)
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),  # ToTensor() divides by 255
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif "r3m" == vision_backbone:
            from r3m import load_r3m

            rep = load_r3m("resnet50")
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose(
                [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
            )  # ToTensor() divides by 255
        else:
            raise NameError("Invalid Model")

        embedding.eval()
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedding_dim * len(self.camera_names) + self.proprio,),
        )

    def observation(self, observation):
        # observation is a dictionary

        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None:
            emb = None

            for k, obs in observation.items():
                inp = self.transforms(Image.fromarray(obs.astype(np.uint8))).reshape(
                    -1, 3, 224, 224
                )
                if self.vision_backbone == "r3m":
                    ## R3M Expects input to be 0-255, preprocess makes 0-1
                    inp *= 255.0
                inp = inp.to(self.device)
                with torch.no_grad():
                    frame_emb = (
                        self.embedding(inp)
                        .view(-1, self.embedding_dim)
                        .to("cpu")
                        .numpy()
                        .squeeze()
                    )
                    if emb is None:
                        emb = frame_emb
                    else:
                        emb = np.concatenate([emb, frame_emb])

            ## IF proprioception add it to end of embedding
            if self.proprio:
                proprio = self.env.unwrapped._get_obs()[: self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()

    def start_finetuning(self):
        self.start_finetune = True


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        width,
        height,
        camera_names,
        device_id=-1,
        depth=False,
        *args,
        **kwargs,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0.0, high=255.0, shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_names = camera_names
        self.depth = depth
        self.device_id = device_id

        # if "v2" in env.spec.id:
        self.get_obs = env._get_obs

    def get_image(self):
        all_camera_names = [
            "corner3",
            "corner",
            "corner2",
            "topview",
            "gripperPOV",
            "behindGripper",
        ]

        if self.camera_names == "default":
            img = self.sim.render(
                width=self.width,
                height=self.height,
                depth=self.depth,
                device_id=self.device_id,
            )
        else:
            # img = self.sim.render(
            #     width=self.width,
            #     height=self.height,
            #     depth=self.depth,
            #     camera_name=self.camera_name,
            #     device_id=self.device_id,
            # )
            img = {
                f"{camera_name}": self.sim.render(
                    height=self.height, width=self.width, camera_name=camera_name
                )
                for camera_name in self.camera_names
                if camera_name in all_camera_names
            }

        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
