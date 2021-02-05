import torch
import torch.nn as nn
from smplx.joint_names import JOINT_NAMES
from smplx import SMPL


class BodyPose(nn.Module):
    def __init__(
        self,
        model,
        keypoint_conf=None,
        dtype=torch.float32,
        device=None,
        model_type="smplx",
        useBodyMeanAngles=True
    ):
        super(BodyPose, self).__init__()
        self.dtype = dtype
        self.device = device
        self.model = model
        self.model_type = model_type
        self.useBodyMeanAngles = useBodyMeanAngles

        # attach SMPL pose tensor as parameter to the layer
        body_pose = torch.zeros(model.body_pose.shape,
                                dtype=dtype, device=device)
        body_pose = nn.Parameter(body_pose, requires_grad=True)
        self.register_parameter("body_pose", body_pose)

    def forward(self, pose_extra):
        pose_in = self.body_pose
        if self.useBodyMeanAngles:
            pose_in = pose_in + pose_extra

        bode_output = self.model(
            body_pose=pose_in
        )

        # store model output for later renderer usage
        self.cur_out = bode_output

        joints = bode_output.joints
        # return a list with invalid joints set to zero
        filtered_joints = joints
        return filtered_joints
