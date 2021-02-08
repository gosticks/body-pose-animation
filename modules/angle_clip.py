import torch
import torch.nn as nn
import numpy as np


class AngleClipper(nn.Module):
    def __init__(
        self,
        device=torch.device('cpu'),
        dtype=torch.float32,
        angle_idx=[24, 10, 9],
        # directions=[-1, 1, 1, 1],
        weight=0.01
    ):
        super(AngleClipper, self).__init__()

        # angles determined based on
        # angles currently only work with SMPL-X since indices may differ
        angles_idx = torch.tensor(
            angle_idx, dtype=torch.int64).to(device=device)
        self.register_buffer("angle_idx",
                             angles_idx)

        # # list of proposed directions
        limit = torch.tensor(np.pi / 2, dtype=dtype).to(device=device)
        self.register_buffer("limit",
                             limit)

        # create buffer for weights
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=dtype).to(device=device)
        )

    def forward(self, pose):

        angles = pose[:, self.angle_idx]

        penalty = angles[torch.abs(angles) > self.limit]

        # get relevant angles
        return penalty.pow(2).sum() * self.weight
