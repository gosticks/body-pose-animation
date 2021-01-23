import torch
import torch.nn.functional as F
import torch.nn as nn
import torchgeometry as tgm


class Transform(nn.Module):
    def __init__(self, dtype, device) -> None:
        super(Transform, self).__init__()

        self.dtype = dtype
        self.device = device

        # init parameters
        translation = torch.rand(3, device=device, dtype=dtype)
        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter("translation", translation)

        orientation = torch.rand((1, 3), device=device, dtype=dtype)
        orientation = nn.Parameter(orientation, requires_grad=True)
        self.register_parameter("orientation", orientation)

    def get_transform_mat(self, with_translate=False):

        transform = tgm.angle_axis_to_rotation_matrix(self.orientation)
        # print(transform.shape)
        if with_translate:
            transform[:, :3, 3] = self.translation
        return transform

    def forward(self, joints):
        R = self.get_transform_mat()
        translation = F.pad(self.translation, (0, 1), value=1)
        return joints @ R + translation
