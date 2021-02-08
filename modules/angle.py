import torch
import torch.nn as nn


class AnglePriorsLoss(nn.Module):
    def __init__(
        self,
        device=torch.device('cpu'),
        dtype=torch.float32,
        angle_idx=[56, 53, 12, 9],
        directions=[1, -1, -1, -1],
        weights=[1.0, 1.0, 1.0, 1.0]
    ):
        super(AnglePriorsLoss, self).__init__()

        # angles determined based on
        # angles currently only work with SMPL-X since indices may differ
        angles_idx = torch.tensor(
            angle_idx, dtype=torch.int64).to(device=device)
        self.register_buffer("angle_idx",
                             angles_idx)

        # list of proposed directions
        angle_directions = torch.tensor(
            directions, dtype=dtype).to(device=device)
        self.register_buffer("angle_directions",
                             angle_directions)

        # create buffer for weights
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=dtype).to(device=device)
        )

    def forward(self, pose):
        # compute direction deviation from expected joint rotation directions,
        # e.g. don't rotate the knee joint forwards. Broken knees are not fun.

        # get relevant angles
        angles = pose[:, self.angle_idx]

        # compute cost based not exponential of angle * direction
        return torch.exp(angles * self.angle_directions).pow(2).sum()
