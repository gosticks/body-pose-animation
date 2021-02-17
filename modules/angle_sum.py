import torch
import torch.nn as nn
import numpy as np


class AngleSumLoss(nn.Module):
    def __init__(
        self,
        device=torch.device('cpu'),
        dtype=torch.float32,
        weight=1,
        # directions=[-1, 1, 1, 1],
        # weight=1
    ):
        super(AngleSumLoss, self).__init__()

        self.has_parameters = False

        self.register_buffer(
            "weight",
            torch.tensor(weight).to(device=device, dtype=dtype)
        )

    def forward(self, pose, joints, points, keypoints, raw_output):
        # get relevant angles
        return pose.pow(2).sum() * self.weight
