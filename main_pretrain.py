# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models_mae
import models_mae_group_channels
import models_mae_temporal
import util.misc as misc
import wandb
from engine_pretrain import train_one_epoch, train_one_epoch_temporal
from util.datasets import build_fmow_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["group_c", "temporal", "vanilla"],
        help="Use channel model",
    )
    parser.add_argument(
        "--model",
        default="mae_vit_small",
        type=str,
        metavar="MODEL",
        help="The name of the model architecture to train. These are defined in models_mae/__init__.py",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="The size of the square-shaped input image",
    )
    parser.add_argument(
        "--patch_size",
        type=str,
        default=8,
        help="The size of the square-shaped patches across the image. Must be a divisor of input_size (input_size % patch_size == 0)",
    )

    parser.add_argument(
        "--print_level",
        type=int,
        default=1,
        help="Print Level (0->3) - Only for MaskedAutoencoderShuntedViT",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--attn_name",
        type=str,
        default="scaled_dot_product",
        help="Attention name to use in transformer block. The following require the --use_xformers flag: 'linformer', 'orthoformer', 'nystrom', 'fourier_mix', 'local'",
        choices=[
            "scaled_dot_product",
            "shunted",
            "linformer",
            "orthoformer",
            "nystrom",
            "fourier_mix",
            "local",
        ],
    )

    parser.add_argument(
        "--use_xformers",
        type=bool,
        action="store_true",
        help="Use xFormers instead of Timm for transformer blocks. Not compatible with --attn_name=shunted",
    )
    parser.set_defaults(use_xformers=False)

    parser.add_argument(
        "--ffn_name",
        type=str,
        default="MLP",
        choices=["MLP", "FusedMLP"],
        help="Type of FFN layer to use. Only supported if --use_xformers is also set.",
    )

    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help="Whether to mask all channels of a spatial location. Only for indp c model",
    )
    # arg for loss, default is mae
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use",
        choices=["mse", "l1"],
    )

    parser.add_argument(
        "--norm_pix_loss",
        type=bool,
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="Learning rate (absolute)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Base LR: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Lower LR bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=40,
        metavar="N",
        help="Defines the epoch where the Warmup Scheduler reaches its maximum value",
    )

    # Dataset parameters
    parser.add_argument(
        "--train_path",
        default="/data2/HDD_16TB/fmow-rgb-preproc/train_128.csv",
        type=str,
        help="Train .csv path",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="rgb",
        choices=["rgb", "temporal", "sentinel", "euro_sat", "naip"],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        type=int,
        nargs="+",
        default=None,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )
    parser.add_argument(
        "--grouped_bands",
        type=int,
        nargs="+",
        action="append",
        default=[],
        help="Bands to group for GroupC mae",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path used for saving trained model checkpoints and logs",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Path used for saving Tensorboard logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training and testing",
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="The path to the checkpoint to resume training from.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="Defines the epoch number to start training from. Useful when resuming training.",
    )

    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Wandb project name, eg: satmae",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="utk-iccv23",
        help="Wandb entity name, eg: utk-iccv23",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="The number of CPU workers to use for the data loader. Generally, this should be set to the number of CPU threads on your machine.",
    )
    parser.add_argument(
        "--pin_mem",
        type=bool,
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--no_pin_mem",
        type=bool,
        action="store_false",
        dest="pin_mem",
        help="Don't pin CPU memory in DataLoader. Could severely slow down training on some systems and datasets.",
    )
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int
    )  # prev default was -1
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("=" * 80)
    print(f"{args}".replace(", ", ",\n"))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #######################################################################################
    print("=" * 80)
    dataset_train = build_fmow_dataset(is_train=True, args=args)
    print(dataset_train)

    global_rank = misc.get_rank()
    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {str(sampler_train)}")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    #######################################################################################
    # define the model
    if args.model_type == "group_c":
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_mae_group_channels.__dict__[args.model](
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands,
            spatial_mask=args.spatial_mask,
            norm_pix_loss=args.norm_pix_loss,
        )
    elif args.model_type == "temporal":
        model = models_mae_temporal.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss
        )
    # non-spatial, non-temporal
    else:
        if args.attn_name == "shunted":
            if "shunted" not in args.model:
                raise ValueError("shunted attention only supported for shunted models")
            sep = "|"
            to_list = lambda x: [int(y) for y in x.split(sep)]
            args.patch_size = to_list(args.patch_size)  # e.g. '16|16' -> [16, 16]

        model = models_mae.__dict__[args.model](**vars(args))
    model.to(device)

    model_without_ddp = model
    print(f"Model = {str(model_without_ddp)}")

    #######################################################################################
    print("=" * 80)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    #######################################################################################
    print("=" * 80)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    #######################################################################################
    print("=" * 80)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_num_params = sum(np.prod(p.size()) for p in model_params)
    print(f"Number of trainable parameters: {model_num_params}")

    #######################################################################################
    print("=" * 80)
    # Set up WandB
    if global_rank == 0 and args.wandb is not None:
        wandb.init(
            project=args.wandb,
            entity=args.wandb_entity,
            group=args.model,
            job_type="pretrain",
        )
        wandb.config.update(args)
        wandb.config.update({"num_params": model_num_params})
        wandb.watch(model)

    # Logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    #######################################################################################
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.model_type == "temporal":
            train_stats = train_one_epoch_temporal(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )
        else:
            train_stats = train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            try:
                if args.wandb is not None:
                    wandb.log(log_stats)
            except ValueError as e:
                traceback.print_exc()
                print(f"Failed to log to wandb: {e}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
