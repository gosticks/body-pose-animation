import torch
import torch.nn as nn


class AnglePriorsLoss(nn.Module):
    def __init__(
        self,
        device=torch.device('cpu'),
        dtype=torch.float32,
        angle_idx=[56, 53, 12, 9, 37, 40],
        directions=[1, 1, -1, -1, 1, -1],
        weight=1,
        weights=[0.8, 0.8, 0.7, 0.7, 0.3, 0.3]
    ):
        super(AnglePriorsLoss, self).__init__()

        self.has_parameters = False

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

        self.register_buffer(
            "global_weight",
            torch.tensor(weight, dtype=dtype).to(device=device)
        )

    def forward(self, pose, joints, points, keypoints, raw_output):
        # compute direction deviation from expected joint rotation directions,
        # e.g. don't rotate the knee joint forwards. Broken knees are not fun.

        # get relevant angles
        angles = pose[:, self.angle_idx]

        # compute cost based not exponential of angle * direction
        return (torch.exp(angles * self.angle_directions) * self.weights).pow(2).sum() * self.global_weight
