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
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        default=None,
        choices=["group_c", "temporal", "vanilla"],
        help="Use channel model",
    )
    parser.add_argument(
        "--model",
        default="mae_vit_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=128, type=int, help="images input size")
    parser.add_argument("--patch_size", default=16, type=str, help="images input size")
    parser.add_argument(
        "--attn_name",
        default="scaled_dot_product",
        # Options: "shunted", "orthoformer", "random", "nystrom", "global", "local", "linformer", "pooling", "fourier_mix", "scaled_dot_product"
        type=str,
        help="attention name to use in transformer block",
    )
    parser.add_argument(
        "--print_level",
        default=1,
        type=int,
        help="Print Level (0->3) - Only for MaskedAutoencoderShuntedViT",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )
    parser.add_argument(
        "--ffn_name",
        default="MLP",
        # Options: "MLP", "FusedMLP"
        type=str,
        help="ffn name to use in transformer block",
    )
    parser.add_argument(
        "--use-xformers",
        action="store_true",
        help="Use xformers instead of timm for transformer",
    )
    parser.set_defaults(use_xformers=False)

    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help="Whether to mask all channels of a spatial location. Only for indp c model",
    )
    # arg for loss, default is mae
    parser.add_argument(
        "--loss",
        default="mse",
        type=str,
        help="Loss function to use (mse or mae)",
    )

    parser.add_argument(
        "--norm_pix_loss",
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
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
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
        default=None,
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Wandb project name, eg: sentinel_pretrain",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="utk-iccv23",
        help="Wandb entity name, eg: utk-iccv23",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int
    )  # prev default was -1
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
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
    batch_size_eff = args.batch_size * args.accum_iter * misc.get_world_size()

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % batch_size_eff)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_size_eff / 256

    print("base lr: %.2e" % (args.lr * 256 / batch_size_eff))
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

    print("=" * 80)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_num_params = sum(np.prod(p.size()) for p in model_params)
    print(f"Trainable parameters: {model_num_params}")

    #######################################################################################
    # Set up output directory, checkpointing, and logging
    print("=" * 80)

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir = [
            "out",
            args.model,
            f"xformers-{args.attn_name}-{args.ffn_name}"
            if args.use_xformers
            else f"{args.attn_name}",
            f"i{args.input_size}-p{args.patch_size}-mr{args.mask_ratio}",
            f"e{args.epochs}-we{args.warmup_epochs}-lr{args.lr}",
            f"b{args.batch_size}-a{args.accum_iter}",
            f"{args.loss_type}{'-normpix' if args.norm_pix_loss else ''}",
        ]

    log_writer = None
    if global_rank == 0:
        if args.wandb is not None:
            wandb.init(
                project=args.wandb,
                entity=args.wandb_entity,
                group=args.model,
                job_type="pretrain",
            )
            wandb.config.update(args)
            wandb.config.update({"num_params": model_num_params})
            wandb.watch(model)
        else:
            print("INFO: Not using wandb.")

        # Logging
        if args.output_dir is not None:
            output_dir_tb = os.path.join(args.output_dir, "tensorboard")
            log_writer = SummaryWriter(log_dir=output_dir_tb)
            print(f"INFO: Tensorboard log path: {output_dir_tb}")
        else:
            print("INFO: Not logging to tensorboard.")

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
                os.path.join(args.output_dir, "log.jsonl"), mode="a", encoding="utf-8"
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
    main(args)
