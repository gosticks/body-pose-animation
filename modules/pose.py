from renderer import Renderer
from utils.mapping import get_mapping_arr
from smplx.body_models import SMPLX
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from smplx import SMPL
from tqdm import tqdm
import torchgeometry as tgm


class BodyPose(nn.Module):
    def __init__(
        self,
        model: SMPL,
        dtype=torch.float32,
        device=None,
    ):
        super(BodyPose, self).__init__()
        self.dtype = dtype
        self.device = device
        self.model = model

        # create valid joint filter
        filter = self.get_joint_filter()
        self.register_buffer("filter", filter)

        # attach SMPL pose tensor as parameter to the layer
        body_pose = torch.zeros(model.body_pose.shape,
                                dtype=dtype, device=device)
        body_pose = nn.Parameter(body_pose, requires_grad=True)
        self.register_parameter("pose", body_pose)

    def get_joint_filter(self):
        """OpenPose and SMPL do not have fully matching joint positions,
            nullify joints not matching between the two. Therefore only matching joints will be
            affected by the optimization
        Args:
            joints ([type]): a full list of SMPL joints.
        """

        # create a list with 1s for used joints and 0 for ignored joints
        mapping = get_mapping_arr()
        filter = torch.zeros(
            (len(mapping), 3), dtype=self.dtype, device=self.device)
        for index, valid in enumerate(mapping > -1):
            if valid:
                filter[index] += 1

        return filter

    def forward(self):
        bode_output = self.model(
            body_pose=self.pose
        )
        # store model output for later renderer usage
        self.cur_out = bode_output

        joints = bode_output.joints

        # return a list with invalid joints set to zero
        return joints * self.filter.unsqueeze(0)


def train_pose(
    model: SMPL,
    keypoints,
    keypoint_conf,
    camera,
    loss_layer=torch.nn.MSELoss(),
    learning_rate=1e-2,
    device=torch.device('cpu'),
    dtype=torch.float32,
    renderer: Renderer = None,
    optimizer=None,
    iterations=25
):

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)
    keypoints_conf = torch.tensor(keypoint_conf).to(device)

    print("setup body pose...")

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device).to(device)

    if optimizer is None:
        optimizer = torch.optim.LBFGS(pose_layer.parameters(), learning_rate)
        # optimizer = torch.optim.Adam(pose_layer.parameters(), learning_rate)

    pbar = tqdm(total=iterations)

    def predict():
        # return joints based on current model state
        body_joints = pose_layer()

        # compute homogeneous coordinates and project them to 2D space
        # TODO: create custom cost function
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()
        return loss_layer(points, keypoints)

    def optim_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = predict()

        if loss.requires_grad:
            loss.backward()
        return loss

    running_loss = 0.0

    for t in range(iterations):
        optimizer.step(optim_closure)

        # LBFGS does not return the result, therefore we should rerun the model to get it
        pred = predict()
        loss = optim_closure()

        # compute loss
        cur_loss = loss.item()

        pbar.set_description("Error %f" % cur_loss)
        pbar.update(1)

        if renderer is not None:
            renderer.render_model(model, pose_layer.cur_out, keep_pose=True)

    pbar.close()
    print("Final result:", loss.item())
