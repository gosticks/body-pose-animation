import yaml
import torch
import math
import time
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint, get_named_joints
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn

# Camera module is heavily inspired by the SMPLify implementation


class Camera(nn.Module):
    def __init__(
        self,
        rotation=None,
        translation=None,
        fx=None,
        fy=None,
        center=None,
        dtype=torch.float32,
        device=None,
        batch_size=1
    ):
        super(Camera, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        self.register_buffer("fx", torch.Tensor([fx], device=device))
        self.register_buffer("fy", torch.Tensor([fy], device=device))
        cam_intr = torch.zeros(
            [batch_size, 2, 2], dtype=dtype, device=device)
        cam_intr[:, 0, 0] = self.fx
        cam_intr[:, 1, 1] = self.fy

        self.register_buffer("cam_intr", cam_intr)
        self.register_buffer("center", center)

        translation = nn.Parameter(
            translation, requires_grad=True)
        self.register_parameter("translation", translation)

        rotation = nn.Parameter(
            rotation,
            requires_grad=True
        )
        self.register_parameter("rotation", rotation)

    def forward(self, points):
        transform = torch.cat([
            F.pad(self.rotation, (0, 0, 0, 1), "constant", value=0),
            F.pad(
                self.translation.unsqueeze(dim=-1),
                (0, 0, 0, 1),
                "constant", value=1)
        ], dim=2)

        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=self.device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [self.cam_intr, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points
