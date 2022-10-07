import torch
import numpy as np


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
