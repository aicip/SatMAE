import glob
import os
import random
import re
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from PIL import Image

import models_mae

# Mean and Std of fmow dataset
# image_mean = np.array([0.4182007312774658, 0.4214799106121063, 0.3991275727748871])
# image_std = np.array([0.28774282336235046, 0.27541765570640564, 0.2764017581939697])

image_mean = np.array([0.40558367, 0.43378946, 0.43175863])
image_std = np.array([0.19208308, 0.19136319, 0.19783947])


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
    print("=" * 80)
    checkpoint_folder = os.path.join(chkpt_basedir, chkpt_dir)

    try:
        if chkpt_name is None:
            # List the directory and find the checkpoint with the highest epoch
            checkpoint_list = os.listdir(checkpoint_folder)
            checkpoint_list = [x for x in checkpoint_list if x.endswith(".pth")]
            checkpoint_list.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            chkpt_name = checkpoint_list[-1]
    except FileNotFoundError as e:
        print("Checkpoint folder not found: ", checkpoint_folder)
        print(f"Checkpoint basedir: {chkpt_basedir}")
        print("Did you mean any of these?")
        # print only folders that contain .pth files
        potential_folders = []
        for folder in glob.glob(f"{chkpt_basedir}/**/*", recursive=True):
            if len(glob.glob(f"{folder}/*.pth")) > 0:
                # also print time last modified in time ago format
                last_modified = os.path.getmtime(folder)
                time_agos = [
                    "sec",
                    "min",
                    "hrs",
                    "days",
                    "wks",
                    "mts",
                    "yrs",
                ]
                potential_folders.append((folder, last_modified))

        potential_folders.sort(key=lambda x: x[1], reverse=True)

        for folder, last_modified in potential_folders:
            last_modified = time.time() - last_modified
            time_ago = "some time"
            for time_ago_ in time_agos:
                time_ago = time_ago_
                if last_modified < 60:
                    break
                last_modified = last_modified / 60

            last_modified = f"{last_modified:.1f} {time_ago} ago"
            folderpath_clean = folder.replace(chkpt_basedir, "").strip("/")
            print(f" - {folderpath_clean:<100} ({last_modified})")

        raise e

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


def prepare_image(image_uri, model, resample=None):
    """
    :param resample: An optional resampling filter.  This can be
           one `Resampling.NEAREST`, `Resampling.BOX`,
           `Resampling.BILINEAR`, `Resampling.HAMMING`,
           `Resampling.BICUBIC`, `Resampling.LANCZOS`.
    """
    img = Image.open(image_uri)

    img_size = model.input_size
    img_chans = model.input_channels

    img = img.resize((img_size, img_size), resample=resample)
    img = np.array(img) / 255.0

    assert img.shape == (img_size, img_size, img_chans)

    # normalize by dataset mean and std
    img = img - image_mean
    img = img / image_std

    return img


def add_noise(image, noise_type="gaussian", noise_param=0.1):
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_param, image.shape)
    elif noise_type == "poisson":
        noise = np.random.poisson(noise_param, image.shape)
    elif noise_type == "s&p":
        noise = np.random.binomial(1, noise_param, image.shape)
    elif noise_type == "speckle":
        noise = np.random.normal(0, noise_param, image.shape)
    else:
        raise ValueError("Invalid noise type")

    noisy_image = image + noise
    return noisy_image


def run_one_image(img, model, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)

    patch_size = model.patch_size
    channels = model.input_channels

    # print("Patch Size:", patch_size)
    # print("Channels:", channels)

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

    y_mask = y * mask
    # MAE reconstruction pasted with visible patches
    im_paste = im_masked + y_mask

    return x, im_masked, y, im_paste


def show_image(image, ax=None, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip((image * image_std + image_mean) * 255, 0, 255).int())
    ax.set_title(title)
    ax.axis("off")
    return


def plot_comp(
    image,
    models,
    maskseed=None,
    use_noise=None,
    resample=None,
    title=None,
    figsize=12,
    savedir="plots",
    save=False,
):
    """
    :param resample: An optional resampling filter.  This can be
           one `Resampling.NEAREST`, `Resampling.BOX`,
           `Resampling.BILINEAR`, `Resampling.HAMMING`,
           `Resampling.BICUBIC`, `Resampling.LANCZOS`.
    """
    # if model is not an array, make it an array
    if not isinstance(models, dict):
        models = {"model": models}

    fig, axs = plt.subplots(
        len(models), 4, figsize=(figsize, len(models) * figsize / 4)
    )

    if title is not None:
        fig.suptitle(title)

    # make the plt figure larger
    # plt.rcParams["figure.figsize"] = [figsize, figsize]
    for model_i, (model_name, model) in enumerate(models.items()):
        # if img is string
        if isinstance(image, str):
            img = prepare_image(image, model, resample=resample)
        else:
            img = image

        if use_noise is not None:
            img = add_noise(img, noise_type=use_noise[0], noise_param=use_noise[1])

        x, im_masked, y, im_paste = run_one_image(img, model, seed=maskseed)

        imgs = [x[0], im_masked[0], y[0], im_paste[0]]
        titles = [
            "Original",
            "Masked",
            f"{model_name} Reconstruction",
            "Reconstruction + Visible",
        ]

        for i in range(4):
            ax = axs[model_i, i] if len(models) > 1 else axs[i]
            show_image(imgs[i], ax, titles[i])

    plt.tight_layout()
    if save:
        if title is not None:
            # replace any symbols with dashes and spaces with underscores
            # replace symbols with dashes
            save_fname = title.replace("-", "")
            save_fname = re.sub(r"[^\w\s]", "-", save_fname)
            # replace spaces with underscores
            save_fname = re.sub(r"\s+", "_", save_fname)

            # if folder does not exist, create it
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, f"plot_viz_{save_fname}.png"))
        else:
            print("INFO: Skipped saving because title was not provided")
    plt.show()


def plot_comp_many(
    models: dict,
    basedir: str,
    max_img_samples: int = 10,
    num_run_each: int = 1,
    random_walk: bool = False,
    walkseed: Optional[int] = None,  # Only used if random_walk is True
    maskseed: Optional[int] = None,
    use_noise: Optional[tuple] = None,  # ex: ("gaussian", 0.25)
    resample=PIL.Image.Resampling.BICUBIC,
    base_title: Optional[str] = None,
    save=False,
):
    # for img_path in glob.iglob(basedir + '**/*.jpg', recursive=True):
    if walkseed is not None:
        if not random_walk:
            print("WARNING: Walkseed only used if random_walk is True")
        random.seed(walkseed)

    def get_image(path_pattern: str):
        if random_walk:
            for _ in range(max_img_samples):
                yield random.choice(glob.glob(path_pattern, recursive=True))
        else:
            for i, img_path in enumerate(glob.iglob(path_pattern, recursive=True)):
                if i >= max_img_samples:
                    break
                yield img_path

    for img_path in get_image(f"{basedir}/**/*.jpg"):
        for _ in range(num_run_each):
            mseed = int(time.time() * 1000) if maskseed is None else maskseed
            base_fname = os.path.basename(img_path)
            title = (
                f"{base_title} - {base_fname}" if base_title is not None else base_fname
            )
            plot_comp(
                img_path,
                models,
                maskseed=mseed,
                title=title,
                use_noise=None,
                resample=resample,
                save=save,
            )
            if use_noise is not None:
                plot_comp(
                    img_path,
                    models,
                    maskseed=mseed,
                    use_noise=use_noise,
                    title=f"{title} - {use_noise[0]} noise {use_noise[1]}",
                    resample=resample,
                    save=save,
                )
