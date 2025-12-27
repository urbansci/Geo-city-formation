"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: Training engine for Masked Autoencoder (MAE) pre-training.
             Provides the core training loop logic for one epoch with support for
             gradient accumulation, mixed precision training, and distributed training.
"""

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    neptune_run=None,
                    args=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The MAE model to train.
        data_loader (Iterable): DataLoader providing training samples.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch number.
        loss_scaler: Gradient scaler for mixed precision training.
        neptune_run: Neptune run object for logging metrics (optional).
        args: Parsed command-line arguments containing training configurations.

    Returns:
        dict: A dictionary containing the global average of all tracked metrics.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast(enabled = False):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if neptune_run is not None:
            neptune_run["train/epoch"].append(epoch)
            neptune_run["train/lr"].append(lr)
            neptune_run["train/loss"].append(loss.item())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}