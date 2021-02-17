import torch
import torch.nn as nn
import numpy as np


class ChangeLoss(nn.Module):
    def __init__(
        self,
        compare_pose,
        device=torch.device('cpu'),
        dtype=torch.float32,
        weight=1,
        # directions=[-1, 1, 1, 1],
        # weight=1
    ):
        super(ChangeLoss, self).__init__()

        self.has_parameters = False

        self.register_buffer(
            "compare_pose",
            compare_pose.to(device=device, dtype=dtype)
        )

        self.loss = nn.MSELoss(reduce="sum").to(
            device=device, dtype=dtype)

        self.register_buffer(
            "weight",
            torch.tensor(weight).to(device=device, dtype=dtype)
        )

    def forward(self, pose, joints, points, keypoints, raw_output):
        return self.loss(self.compare_pose, pose) * self.weight
