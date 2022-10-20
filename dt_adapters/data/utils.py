import torch
import numpy as np

from transformers import CLIPProcessor, CLIPVisionModel
from torchvision.transforms import transforms as T
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


def preprocess_obs(config, obs):
    action = np.concatenate([obs.joint_velocities, np.array([obs.gripper_open])])

    ll_state_info = [
        np.array(getattr(obs, k)).reshape(-1) for k in config.ll_state_keys
    ]
    ll_state = np.concatenate(ll_state_info)

    image_state = {k: getattr(obs, k) for k in config.image_keys}
    return action, ll_state, image_state


def extract_image_feats(
    img_obs, img_preprocessor, img_encoder, depth_img_preprocessor, depth_img_encoder
):
    all_img_feats = {}
    for k, imgs in img_obs.items():
        if "rgb" in k:
            img_feat = get_image_feats(
                np.array(imgs),
                img_preprocessor,
                img_encoder,
                "clip",
            )
        if "depth" in k:
            img_feat = get_image_feats(
                np.array(imgs),
                depth_img_preprocessor,
                depth_img_encoder,
                "resnet",
            )
        all_img_feats[k] = img_feat
    # all_img_feats = np.concatenate(all_img_feats, axis=-1)
    return all_img_feats


def get_visual_encoders(image_size, device):
    img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    img_encoder.to(device).eval()
    img_preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # for processing depth images
    # self.depth_img_preprocessor = weights.transforms()
    img_transforms = T.Compose(
        [
            T.Lambda(
                lambda images: torch.stack([T.ToTensor()(image) for image in images])
            ),
            T.Resize([image_size]),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Lambda(lambda images: images.numpy()),
        ]
    )
    depth_img_preprocessor = img_transforms
    weights = ResNet50_Weights.DEFAULT
    depth_img_encoder = resnet50(weights=weights)
    depth_img_encoder.to(device).eval()
    return img_preprocessor, img_encoder, depth_img_preprocessor, depth_img_encoder


def get_image_feats(images, img_preprocessor, img_encoder, vision_backbone="clip"):
    # assume images is a numpy array of LxHxWxC
    if len(images.shape) == 3:
        pass
    elif len(images.shape) == 4:
        # apply transform to images first
        # need to reshape into LxCxHxW
        images = images.transpose(0, 3, 1, 2)

    img_feats = None
    # get pretrained image features
    if vision_backbone == "clip":
        # for some reason clip preprocessor needs a list of images
        list_images = [images[i] for i in range(images.shape[0])]
        images = torch.stack(
            list(img_preprocessor(images=list_images, return_tensors="pt").pixel_values)
        ).to("cuda")
        with torch.no_grad():
            outputs = img_encoder(pixel_values=images)
        # last_hidden_state = outputs.last_hidden_state
        img_feats = outputs.pooler_output.detach().cpu().numpy()
    elif vision_backbone == "resnet":
        preprocessed_img = img_preprocessor(images)
        # if grayscale, we need to repeat to make it look like RGB for pretrained resnet
        if preprocessed_img.shape[1] == 1:
            preprocessed_img = torch.Tensor(preprocessed_img).repeat(1, 3, 1, 1)

        preprocessed_img = preprocessed_img.cuda()
        img_feats = img_encoder(preprocessed_img).detach().cpu().numpy()
    else:
        raise NotImplementedError()

    return img_feats
