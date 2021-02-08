
from camera_estimation import TorchCameraEstimate
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, sin
from model import *
from dataset import *


class TransformCamera(nn.Module):
    def __init__(
        self,
        transform_mat: torch.Tensor,
        dtype=torch.float32,
        device=None,
    ):
        super(TransformCamera, self).__init__()

        self.dtype = dtype
        self.device = device

        self.register_buffer("trans", transform_mat.to(
            device=device, dtype=dtype))

    def forward(self, points):
        proj_points = self.trans @ points.reshape(-1, 4, 1)
        proj_points = proj_points.reshape(1, -1, 4)[:, :, :2] * 1
        proj_points = F.pad(proj_points, (0, 1, 0, 0), value=0)
        return proj_points


class IntrinsicsCamera(nn.Module):
    def __init__(
        self,
        transform_mat: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_trans_rot: torch.Tensor,
        dtype=torch.float32,
        device=None
    ):
        super(IntrinsicsCamera, self).__init__()

        self.dtype = dtype
        self.device = device

        self.register_buffer("cam_int", camera_intrinsics.to(
            device=device, dtype=dtype))
        self.register_buffer("cam_trans_rot", camera_trans_rot.to(
            device=device, dtype=dtype))
        self.register_buffer("trans", transform_mat.to(
            device=device, dtype=dtype))

    def forward(self, points):
        proj_points = self.cam_int[:3, :3] @ self.cam_trans_rot[:3,
                                                                :] @ self.trans @ points.reshape(-1, 4, 1)
        result = proj_points.squeeze(2)
        denomiator = torch.zeros(
            points.shape[1], 3, device=self.device, dtype=self.dtype)
        for i in range(points.shape[1]):
            denomiator[i, :] = result[i, 2]
        result = result/denomiator
        result[:, 2] = 0
        return result


class SimpleCamera(nn.Module):
    def from_estimation_cam(cam: TorchCameraEstimate, use_intrinsics=False, device=None, dtype=None):
        """utility to create camera module from estimation camera

        Args:
            cam (TorchCameraEstimate): pre trained estimation camera
        """
        cam_trans, cam_int, cam_params = cam.get_results(
            device=device, dtype=dtype, visualize=True)

        cam_layer = None

        if use_intrinsics:
            cam_layer = IntrinsicsCamera(
                transform_mat=cam_trans,
                camera_intrinsics=cam_int,
                camera_trans_rot=cam_params,
                device=device,
                dtype=dtype,
            )
        else:
            cam_layer = TransformCamera(
                transform_mat=cam_trans,
                device=device,
                dtype=dtype,
            )

        return cam_layer, cam_trans, cam_int, cam_params
