from smplx.body_models import SMPLX
import torch
import torch.nn.functional as F
import torch.nn as nn
from smplx import SMPL


class BodyPose(nn.Module):
    def __init__(
        self,
        model: SMPL,
        dtype=torch.float32,
        device=None,
    ):
        super(BodyPose, self).__init__()
        self.register_parameter("pose", model.body_pose)

    def forward(self):
        return self.pose
