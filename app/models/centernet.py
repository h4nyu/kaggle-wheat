import torch
import numpy as np
import typing as t
import torchvision
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from .modules import ConvBR2d, SENextBottleneck2d
from .bifpn import BiFPN, FP
from app.dataset import Targets, Target
from scipy.stats import multivariate_normal
from .utils import box_cxcywh_to_xyxy
from app.utils import plot_heatmap
from pathlib import Path
import albumentations as albm

Outputs = t.TypedDict(
    "Outputs", {"heatmap": Tensor, "box_size": Tensor,}  # [B, 1, W, H]  # [B, 2, W, H]
)


class VisualizeHeatmap:
    def __init__(self, output_dir: Path, prefix: str = "",) -> None:
        self.prefix = prefix
        self.output_dir = output_dir

    def __call__(self, outputs: Outputs, targets: "CriterinoTarget") -> None:
        plot_heatmap(
            targets["mask"][-1][0].detach().cpu(),
            self.output_dir.joinpath(f"{self.prefix}-tgt-mask.png"),
        )
        plot_heatmap(
            outputs["heatmap"][-1][0].detach().cpu(),
            self.output_dir.joinpath(f"{self.prefix}-src.png"),
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        alpha = self.alpha
        beta = self.beta
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        neg_loss = (
            -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
        )
        loss = (pos_loss + neg_loss).sum()
        num_pos = pos_mask.sum().clamp(1)
        return loss / num_pos.float()


def gaussian_2d(shape: t.Any, sigma: float = 1) -> np.ndarray:
    m, n = int((shape[0] - 1.0) / 2.0), int((shape[1] - 1.0) / 2.0)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class Backbone(nn.Module):
    def __init__(self, name: str, out_channels: int) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        if name == "resnet34" or name == "resnet18":
            num_channels = 512
        else:
            num_channels = 2048
        self.layers = list(self.backbone.children())[:-2]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=i, out_channels=out_channels, kernel_size=1,)
                for i in [64, 64, 128, 256, 512]
            ]
        )

    def forward(self, x: Tensor) -> FP:
        internal_outputs = []
        for layer in self.layers:
            x = layer(x)
            internal_outputs.append(x)
        _, p3, _, p4, _, p5, p6, p7 = internal_outputs
        return (
            self.projects[0](p3),
            self.projects[1](p4),
            self.projects[2](p5),
            self.projects[3](p6),
            self.projects[4](p7),
        )


class HardHeatMap(nn.Module):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, boxes: Tensor) -> Tensor:
        img = torch.zeros((1, 1, self.w, self.h), dtype=torch.float32).to(boxes.device)
        b, _ = boxes.shape
        if b == 0:
            return img
        cx, cy = (boxes[:, 0] * self.w).long(), (boxes[:, 1] * self.h).long()
        img[:, :, cx, cy] = 1.0
        return img


class SoftHeatMap(nn.Module):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h
        mount = gaussian_2d((64, 64), sigma=7)
        self.mount = torch.tensor(mount, dtype=torch.float32).view(
            1, 1, mount.shape[0], mount.shape[1]
        )

    def forward(self, boxes: Tensor) -> Tensor:
        img = torch.zeros((1, 1, self.w, self.h), dtype=torch.float32).to(boxes.device)
        b, _ = boxes.shape
        if b == 0:
            return img
        xyxy = box_cxcywh_to_xyxy(boxes)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * self.w
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * self.w
        xyxy = xyxy.long()
        sizes = xyxy[:, [2, 3]] - xyxy[:, [0, 1]]
        for box, size in zip(xyxy, sizes):
            x, y = box[0], box[1]
            w, h = size[0], size[1]
            mount = F.interpolate(self.mount, size=(w, h)).to(boxes.device)
            mount = torch.max(mount, img[:, :, x : x + w, y : y + h])  # type: ignore
            img[:, :, x : x + w, y : y + h] = mount  # type: ignore
        return img


class PreProcess(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.heatmap = HardHeatMap(w=1024 // 4, h=1024 // 4)

    def forward(
        self, batch: t.Tuple[Tensor, t.List[Target]]
    ) -> t.Tuple[Tensor, "CriterinoTarget"]:
        images, targets = batch
        heatmap = torch.cat([self.heatmap(t["boxes"]) for t in targets], dim=0)
        mask = (heatmap > 0.5).long()
        return images, dict(heatmap=heatmap, mask=mask)


CriterinoTarget = t.TypedDict("CriterinoTarget", {"heatmap": Tensor, "mask": Tensor,})


class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()

    def forward(self, src: Outputs, tgt: CriterinoTarget) -> Tensor:
        heatmap = src["heatmap"]
        b, _, _, _ = heatmap.shape
        mask = tgt["mask"]
        return self.focal_loss(heatmap, mask) / b


class Reg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(SENextBottleneck2d(in_channels, channels))

        self.out = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.conv(x)
        x = self.out(x)
        return x


class ToBoxes(nn.Module):
    def __init__(self, thresold: float) -> None:
        super().__init__()
        self.thresold = thresold

    def forward(self, heatmaps: Tensor, sizes: Tensor) -> t.List[t.Tuple[Tensor, Tensor]]:
        device = heatmaps.device
        kp_maps = (F.max_pool2d(heatmaps, 3, stride=1, padding=1) == heatmaps) & (
            heatmaps > self.thresold
        )
        batch_size, _, width, height = heatmaps.shape
        original_size = torch.tensor([width, height], dtype=torch.float32).to(device)
        targets: t.List[t.Tuple[Tensor, Tensor]] = []
        for hm, kp_map, size_map in zip(heatmaps.squeeze(1), kp_maps.squeeze(1), sizes):
            pos = kp_map.nonzero()
            confidences = hm[pos[:, 0], pos[:, 1]]
            wh = size_map[:, pos[:, 0], pos[:, 1]]
            cxcy = pos.float() / original_size
            boxes = torch.cat([cxcy, wh], dim=1)
            targets.append((
                confidences,
                boxes,
            ))
        return targets


class Augmention:
    def __call__(self, images: Tensor) -> None:
        transform = albm.Compose([albm.VerticalFlip(), albm.HorizontalFlip(),])


class CenterNet(nn.Module):
    def __init__(
        self, name: str = "resnet18", num_classes: int = 1, num_queries: int = 100
    ) -> None:
        super().__init__()
        channels = 32
        self.backbone = Backbone(name, out_channels=channels)
        self.fpn = nn.Sequential(BiFPN(channels=channels), BiFPN(channels=channels),)
        self.heatmap = Reg(in_channels=channels, out_channels=1)
        self.box_size = Reg(in_channels=channels, out_channels=2)

    def forward(self, x: Tensor) -> Outputs:
        """
        x: [B, 3, W, H]
        """
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmap = self.heatmap(fp[1])
        box_size = self.box_size(fp[1])
        return dict(heatmap=heatmap, box_size=box_size)
