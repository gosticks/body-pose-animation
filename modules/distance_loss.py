import torch
import torch.nn.functional as F
import torch.nn as nn
import torchgeometry as tgm


class WeightedMSELoss(nn.Module):
    def __init__(
            self,
            weights,
            dtype=torch.float32,
            device=torch.device('cpu')) -> None:
        super(WeightedMSELoss, self).__init__()

        self.dtype = dtype
        self.device = device
        weights = torch.tensor(weights).to(dtype=dtype, device=device)
        weights = weights.unsqueeze(1).repeat(1, 3)
        self.register_buffer("weights", weights)

    def forward(self, joints, keypoints):
        return torch.sum(((joints - keypoints) * self.weights) ** 2)
