import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import models_mae

# Mean and Std of fmow dataset
image_mean = np.array([0.4182007312774658, 0.4214799106121063, 0.3991275727748871])
image_std = np.array([0.28774282336235046, 0.27541765570640564, 0.2764017581939697])


def prepare_model(
    chkpt_dir,
    chkpt_basedir="../Model_Saving",
    chkpt_name=None,
    arch="mae_vit_base",
    map_location="cpu",
):
    """
    Loads the model from checkpoint

    Arguments:
        chkpt_dir -- Ex: out_mae_vit_small_scaled_dot_product_i128_p16_b512_e200

    Keyword Arguments:
        chkpt_basedir -- Base directory where multiple models checkpoints are stored directories (default: {"../Model_Saving"})
        chkpt_name -- If none, load the latest (default: {None})
        arch -- Model architecture being used (default: {"mae_vit_base"})
        map_location -- Where to load the state dict of model (default: {"cpu"})

    Returns:
        The model
    """
    checkpoint_folder = os.path.join(chkpt_basedir, chkpt_dir)

    if chkpt_name is None:
        # List the directory and find the checkpoint with the highest epoch
        checkpoint_list = os.listdir(checkpoint_folder)
        checkpoint_list = [x for x in checkpoint_list if x.endswith(".pth")]
        checkpoint_list.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        chkpt_name = checkpoint_list[-1]

    if not chkpt_name.endswith(".pth"):
        chkpt_name = f"{chkpt_name}.pth"

    if not chkpt_name.startswith("checkpoint-"):
        chkpt_name = f"checkpoint-{chkpt_name}"

    chkpt_path = os.path.join(checkpoint_folder, chkpt_name)
    print("Loading checkpoint: ", chkpt_path)
    checkpoint = torch.load(chkpt_path, map_location=map_location)

    # build model
    args = vars(checkpoint["args"])
    print("args:", args)
    model = getattr(models_mae, arch)(**args)
    # load model
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    print("Model loaded.")
    return model


def prepare_image(image_uri, model):
    img = Image.open(image_uri)

    img_size = model.input_size
    img_chans = model.input_channels

    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0

    assert img.shape == (img_size, img_size, img_chans)

    # normalize by ImageNet mean and std
    img = img - image_mean
    img = img / image_std

    return img


def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * image_std + image_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def run_one_image(img, model, seed: Optional[int] = None):
    if seed != None:
        torch.manual_seed(seed)

    patch_size = model.patch_size
    channels = model.input_channels

    print("Patch Size:", patch_size)
    print("Channels:", channels)

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    _, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y, p=patch_size, c=channels)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(
        mask, p=patch_size, c=channels
    )  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()
