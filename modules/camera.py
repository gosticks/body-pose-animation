
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, sin
from model import *
from dataset import *


class SimpleCamera(nn.Module):
    def __init__(
        self,
        dtype=torch.float32,
        device=None,
        z_scale=0.5,
        transform_mat=None,
    ):
        super(SimpleCamera, self).__init__()
        self.hasTransform = False
        self.dtype = dtype
        self.device = device

        zs = torch.ones(1, device=device)
        zs *= z_scale
        self.register_buffer("z_scale", zs)

        if transform_mat is not None:
            self.hasTransform = True
            self.register_buffer("trans", transform_mat)

    def forward(self, points):
        if self.hasTransform:
            proj_points = points @ self.trans
        # scale = (points[:, :, 2] / self.z_scale)
        # print(points.shape, scale.shape)
        proj_points = proj_points[:, :, :2] * 1
        proj_points = F.pad(proj_points, (0, 1, 0, 0), value=0)
        return proj_points
