from model import VPoserModel
import torch
import torch.nn as nn
import numpy as np


class BodyPrior(nn.Module):
    def __init__(
        self,
        vmodel: VPoserModel,
        device=torch.device('cpu'),
        dtype=torch.float32,
        # directions=[-1, 1, 1, 1],
        weight=1
    ):
        super(BodyPrior, self).__init__()

        self.has_parameters = True

        self.model = vmodel.model.to(device=device, dtype=dtype)
        latent_pose = vmodel.get_vposer_latent()
        self.register_parameter("latent_pose", latent_pose)

        # create buffer for weights
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=dtype).to(device=device)
        )

    def forward(self, pose, joints, points, keypoints):
        # get relevant angles
        return self.latent_pose.pow(
            2).sum() * self.weight
