from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


class SwinLBackboneFPN(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.body = timm.create_model(
            "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self.in_channels_list = list(self.body.feature_info.channels())
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelP6P7(self.in_channels_list[-1], 256),
        )
        self.out_channels = 256

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.body(images)
        feature_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for idx, feature in enumerate(features):
            if feature.ndim != 4:
                raise RuntimeError(f"Expected 4D feature map, got shape {tuple(feature.shape)}")
            if feature.shape[1] != self.in_channels_list[idx] and feature.shape[-1] == self.in_channels_list[idx]:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            feature_dict[str(idx)] = feature
        return self.fpn(feature_dict)


def build_model(
    num_classes: int,
    image_size: int,
    pretrained_backbone: bool = True,
) -> RetinaNet:
    backbone = SwinLBackboneFPN(pretrained=pretrained_backbone)
    anchor_generator = AnchorGenerator(
        sizes=(
            (20, 25, 32),
            (40, 50, 64),
            (80, 101, 128),
            (161, 203, 256),
            (322, 406, 512),
        ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    head = RetinaNetHead(
        in_channels=backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"

    return RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        head=head,
        min_size=(image_size,),
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )


def build_optimizer(
    model: RetinaNet,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> AdamW:
    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    head_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone_body = name.startswith("backbone.body.")
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")

        if is_backbone_body and use_decay:
            backbone_decay.append(parameter)
        elif is_backbone_body:
            backbone_no_decay.append(parameter)
        elif use_decay:
            head_decay.append(parameter)
        else:
            head_no_decay.append(parameter)

    return AdamW(
        [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps == warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def optimizer_lrs(optimizer: AdamW) -> dict[str, float]:
    return {
        "train/lr_backbone": optimizer.param_groups[0]["lr"],
        "train/lr_head": optimizer.param_groups[2]["lr"],
    }
