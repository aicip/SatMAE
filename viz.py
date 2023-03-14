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
from pytorch_msssim import MS_SSIM, SSIM, ms_ssim, ssim
from torchvision import transforms as T

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
        print_checkpoint_folders(chkpt_basedir)
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
    if "print_level" in args:
        args["print_level"] = 0
    print("args:", args)
    try:
        model = getattr(models_mae, arch)(**args)
    except AssertionError as e:
        print("Error: ", e)
        return None
    # load model
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    print("Model loaded.")
    return model


def print_checkpoint_folders(chkpt_basedir):
    print("Available checkpoint folders:")
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


def prepare_image(image_uri, model, random_crop=False, crop_seed=None, resample=None):
    """
    :param resample: An optional resampling filter.  This can be
           one `Resampling.NEAREST`, `Resampling.BOX`,
           `Resampling.BILINEAR`, `Resampling.HAMMING`,
           `Resampling.BICUBIC`, `Resampling.LANCZOS`.
    """
    img = Image.open(image_uri)

    img_size = model.input_size
    img_chans = model.input_channels

    # random crop
    if random_crop:
        if crop_seed is not None:
            torch.manual_seed(crop_seed)
        img = T.RandomResizedCrop(img_size, scale=(0.2, 0.8))(img)

    img = img.resize((img_size, img_size), resample=resample)
    img = np.array(img) / 255.0
    img = (img - image_mean) / image_std

    assert img.shape == (img_size, img_size, img_chans)

    return img


def add_noise(image, noise_type="gaussian", noise_param=0.1):
    # if noise type is random, randomly choose one of the other types
    if noise_type == "random":
        noise_type = np.random.choice(["gaussian", "poisson", "s&p"])

    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_param, image.shape)
    elif noise_type == "poisson":
        noise = np.random.poisson(noise_param, image.shape)
    elif noise_type == "s&p":
        noise = np.random.binomial(1, noise_param, image.shape)
    else:
        raise ValueError("Invalid noise type")

    return image + noise


def add_noise_torch(image, noise_type="gaussian", noise_param=0.1):
    # create a tensor the same size as the image
    if noise_type == "gaussian":
        noise = torch.randn_like(image) * noise_param
    elif noise_type == "poisson":
        noise = torch.poisson(torch.ones_like(image) * noise_param)
    elif noise_type == "s&p":
        noise = torch.bernoulli(torch.ones_like(image) * noise_param)
    else:
        raise ValueError("Invalid noise type")

    return image + noise.to(image.device)


@torch.no_grad()
def run_one_image(img, model, seed: Optional[int] = None, device=None):
    if "patch_size" not in model.__dict__:  # for shunted models
        patch_size = model.patch_sizes[-1]
    else:
        patch_size = model.patch_size
    channels = model.input_channels

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    if "mask_ratio" in model.__dict__:
        mask_ratio = model.mask_ratio
    else:
        mask_ratio = 0.75
        print(
            f"WARN: mask_ratio not found in model config. Defaulting to {mask_ratio}."
        )

    xf = x.float()
    if device is not None:
        xf = xf.to(device)
    _, y, mask = model(xf, mask_ratio=mask_ratio, mask_seed=seed)

    y = model.unpatchify(y, p=patch_size, c=channels)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    if "num_stages" in model.__dict__:
        stage = model.num_stages - 1
        patch_embed = getattr(model, f"patch_embed{stage + 1}")
    else:
        patch_embed = model.patch_embed
    mask = mask.unsqueeze(-1).repeat(
        1, 1, patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(
        mask, p=patch_size, c=channels
    )  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    # Convert to channel last
    x = torch.einsum("nchw->nhwc", x)

    # Un-normalize
    x = (x * image_std) + image_mean
    y = (y * image_std) + image_mean

    # masked image
    im_masked = x * (1 - mask)

    y_mask = y * mask
    # MAE reconstruction pasted with visible patches
    im_paste = im_masked + y_mask

    return x, im_masked, y, im_paste


def show_image(image, ax=None, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    # if needed conver to int 0-255
    if image.dtype != np.uint8:
        image = torch.clip((image) * 255, 0, 255).int()

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    # show min and max values at the bottom
    # vmin = image.min().item()
    # vmax = image.max().item()
    # ax.text(
    #     0,
    #     0,
    #     f"{vmin} - {vmax}",
    #     color="white",
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    #     transform=ax.transAxes,
    # )

    return


def plot_comp(
    image,
    models,
    maskseed=None,
    use_noise=None,
    use_random_crop=False,
    comp_metric="ssd",
    resample=None,
    title=None,
    figsize=12,
    savedir="./plots",
    save=False,
    show=True,
    device=None,
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
        len(models),
        5,
        figsize=(figsize, len(models) * figsize / 4),
    )

    if title is not None:
        fig.suptitle(title)

    cropseed = np.random.randint(1000000) if use_random_crop else None

    for model_i, (model_name, model) in enumerate(models.items()):
        # if img is string
        if isinstance(image, str):
            img = prepare_image(
                image,
                model,
                resample=resample,
                random_crop=use_random_crop,
                crop_seed=cropseed,
            )
        else:
            img = image

        if use_noise is not None:
            img = add_noise(img, noise_type=use_noise[0], noise_param=use_noise[1])

        x, im_masked, y, im_paste = run_one_image(
            img, model, seed=maskseed, device=device
        )

        diff = torch.abs(x[0] - y[0])

        comp_metric = comp_metric.lower()
        if comp_metric == "mse":
            diff_m = torch.mean((x[0] - y[0]) ** 2).item()
        elif comp_metric == "l1":
            diff_m = torch.mean(torch.abs(x[0] - y[0])).item()

        elif comp_metric == "ssd":
            diff_m = torch.sum((x[0] - y[0]) ** 2).item()
        elif comp_metric == "sad":
            diff_m = torch.sum(torch.abs(x[0] - y[0])).item()

        elif comp_metric == "ssim":
            diff_m = ssim(x, y, data_range=1, size_average=True).item()
        elif comp_metric == "msssim":
            diff_m = ms_ssim(x, y, data_range=1, size_average=True).item()

        else:
            raise ValueError(f"Unknown metric {comp_metric}")

        imgs = [x[0], im_masked[0], y[0], diff, im_paste[0]]
        titles = [
            "Original",
            "Masked",
            f"{model_name}",
            f"{comp_metric.upper()} = {diff_m:.3f}",
            "Reconstruction + Visible",
        ]

        for i in range(5):
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

            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, f"plot_viz_{save_fname}.png"))
        else:
            print("INFO: Skipped saving because title was not provided")

    if show:
        plt.show()

    # Return the plot as a pixel array
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def plot_comp_many(
    models: dict,
    basedir: str,
    max_img_samples: int = 10,
    num_run_each: int = 1,
    random_walk: bool = False,
    walkseed: Optional[int] = None,  # Only used if random_walk is True
    maskseed: Optional[int] = None,
    use_noise: Optional[tuple] = None,  # ex: ("gaussian", 0.25)
    use_random_crop: bool = False,
    num_random_crop: int = 1,
    resample=PIL.Image.Resampling.BICUBIC,
    comp_metric="ssd",
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
                comp_metric=comp_metric,
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
                    comp_metric=comp_metric,
                    save=save,
                )
            if use_random_crop:
                for _ in range(num_random_crop):
                    plot_comp(
                        img_path,
                        models,
                        maskseed=mseed,
                        use_random_crop=True,
                        title=f"{title} - Random Resize Crop",
                        resample=resample,
                        comp_metric=comp_metric,
                        save=save,
                    )
