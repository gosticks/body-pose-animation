
from camera_estimation import TorchCameraEstimate
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, sin
from model import *
from dataset import *


class SimpleCamera(nn.Module):
    def __init__(
        self,
        dtype=torch.float32,
        device=None,
        transform_mat=None,
        camera_intrinsics=None,
        camera_trans_rot=None
    ):
        super(SimpleCamera, self).__init__()
        self.hasTransform = False
        self.hasCameraTransform = False
        self.dtype = dtype
        self.device = device
        self.model_type = "smplx"

        if camera_intrinsics is not None:
            self.hasCameraTransform = True
            self.register_buffer("cam_int", camera_intrinsics)
            self.register_buffer("cam_trans_rot", camera_trans_rot)
            self.register_buffer("trans", transform_mat)
            # self.register_buffer("disp_trans", camera_trans_rot)
        elif transform_mat is not None:
            self.hasTransform = True
            self.register_buffer("trans", transform_mat)
            # self.register_buffer("disp_trans", transform_mat)

    def from_estimation_cam(cam: TorchCameraEstimate, device=None, dtype=None):
        """utility to create camera module from estimation camera

        Args:
            cam (TorchCameraEstimate): pre trained estimation camera
        """
        cam_trans, cam_int, cam_params = cam.get_results(
            device=device, dtype=dtype)

        return SimpleCamera(
            dtype,
            device,
            transform_mat=cam_trans,
              camera_intrinsics=cam_int, camera_trans_rot=cam_params
        ), cam_trans, cam_int, cam_params

    def forward(self, points):
        if self.hasTransform:
            proj_points = self.trans @ points.reshape(-1, 4, 1)
            proj_points = proj_points.reshape(1, -1, 4)[:, :, :2] * 1
            proj_points = F.pad(proj_points, (0, 1, 0, 0), value=0)
            return proj_points
        if self.hasCameraTransform:
            proj_points = self.cam_int[:3, :3] @ self.cam_trans_rot[:3,
                                                                    :] @ self.trans @ points.reshape(-1, 4, 1)
            result = proj_points.squeeze(2)
            denomiator = torch.zeros(points.shape[1], 3)
            for i in range(points.shape[1]):
                denomiator[i, :] = result[i, 2]
            result = result/denomiator
            result[:, 2] = 0
            return result

        # scale = (points[:, :, 2] / self.z_scale)
        # print(points.shape, scale.shape)
