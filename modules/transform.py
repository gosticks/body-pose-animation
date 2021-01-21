import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class Transform(nn.Module):
    def __init__(self, dtype, device) -> None:
        super(Transform, self).__init__()

        self.dtype = dtype
        self.device = device

        # init parameters
        translation = torch.zeros(3, device=device, dtype=dtype)
        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter("translation", translation)

        orientation = torch.ones((1, 3), device=device, dtype=dtype)
        orientation = nn.init.xavier_uniform_(
            orientation, gain=1.0)
        orientation = orientation.clamp(-math.pi * 0.25, math.pi * 0.25)
        orientation = nn.Parameter(orientation.squeeze(), requires_grad=True)
        self.register_parameter("orientation", orientation)

        # self.roll = torch.randn(
        #     1,  device=device, dtype=dtype,  requires_grad=True)
        # self.yaw = torch.randn(
        #     1,  device=device, dtype=dtype,  requires_grad=True)
        # self.pitch = torch.randn(
        #     1,  device=device, dtype=dtype,  requires_grad=True)

        # init addition buffers
        tensor_0 = torch.zeros(1,  device=device, dtype=dtype)
        self.register_buffer("tensor_0", tensor_0)
        tensor_1 = torch.ones(1,  device=device, dtype=dtype)
        self.register_buffer("tensor_1", tensor_1)

    def get_transform_mat(self):
        tensor_1 = self.tensor_1.squeeze()
        tensor_0 = self.tensor_0.squeeze()
        roll = self.orientation[0]
        pitch = self.orientation[1]
        yaw = self.orientation[2]

        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RX, RY)
        R = torch.mm(R, RZ)
        # R = torch.mm(RZ, RY)
        #R = torch.mm(R, RX)
        return R

    def forward(self, joints):
        R = self.get_transform_mat()

        return joints @ R + self.translation
