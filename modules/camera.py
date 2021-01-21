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


# class Camera(nn.Module):
#     def __init__(
#         self,
#         fx=None,
#         fy=None,
#         center=None,
#         plain_depth=0,
#         dtype=torch.float32,
#         device=None,
#     ):
#         super(Camera, self).__init__()

#         self.dtype = dtype
#         self.device = device

#         self.register_buffer("fx", torch.Tensor([fx], device=device))
#         self.register_buffer("fy", torch.Tensor([fy], device=device))
#         camera_intrinsics = torch.zeros(
#             [2, 2], dtype=dtype, device=device)
#         camera_intrinsics[0, 0] = self.fx
#         camera_intrinsics[1, 1] = self.fy

#         self.register_buffer("camera_intrinsics",
#                              torch.inverse(camera_intrinsics))
#         self.register_buffer("center", center)

#     def forward(self, joints):
#         # translate to homogeneous coordinates
#         homog_coord = torch.ones(
#             list(joints.shape)[:-1] + [1],
#             dtype=self.dtype,
#             device=self.device)
#         # Convert the points to homogeneous coordinates
#         projected_points = torch.cat([joints, homog_coord], dim=-1)

#         img_points = torch.div(projected_points[:, :, :2],
#                                projected_points[:, :, 2].unsqueeze(dim=-1))
#         img_points = torch.einsum('bki,bji->bjk', [self.camera_intrinsics, img_points]) \
#             + self.center.unsqueeze(dim=1)
#         return img_points

class CameraProjSimple(nn.Module):
    def __init__(
        self,
        dtype=torch.float32,
        device=None,
        z_scale=0.5
    ):
        super(CameraProjSimple, self).__init__()

        self.dtype = dtype
        self.device = device

        zs = torch.ones(1, device=device)
        zs *= z_scale
        print(zs)
        self.register_buffer("z_scale", zs)

    def forward(self, points):
        proj_points = torch.mul(
            points[:, :, :2], points[:, :, 2] / self.z_scale)
        proj_points = F.pad(proj_points, (0, 1, 0, 0), value=0)
        return proj_points
