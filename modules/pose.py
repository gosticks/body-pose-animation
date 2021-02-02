from modules.priors import SMPLifyAnglePrior
from model import VPoserModel
from modules.camera import SimpleCamera
from renderer import Renderer
from utils.mapping import get_mapping_arr, get_named_joint, get_named_joints
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from smplx.joint_names import JOINT_NAMES
from smplx import SMPL
from tqdm import tqdm
import torchgeometry as tgm
from human_body_prior.tools.model_loader import load_vposer


class BodyPose(nn.Module):
    def __init__(
        self,
        model,
        keypoint_conf=None,
        dtype=torch.float32,
        device=None,
        model_type="smplx"

    ):
        super(BodyPose, self).__init__()
        self.dtype = dtype
        self.device = device
        self.model = model
        self.model_type = model_type

        # create valid joint filter
        filter = self.get_joint_filter()
        self.register_buffer("filter", filter)

        # attach SMPL pose tensor as parameter to the layer
        body_pose = torch.zeros(model.body_pose.shape,
                                dtype=dtype, device=device)
        body_pose = nn.Parameter(body_pose, requires_grad=True)
        self.register_parameter("body_pose", body_pose)

    def get_joint_filter(self):
        """OpenPose and SMPL do not have fully matching joint positions,
            nullify joints not matching between the two. Therefore only matching joints will be
            affected by the optimization
        Args:
            joints ([type]): a full list of SMPL joints.
        """

        # create a list with 1s for used joints and 0 for ignored joints
        mapping = get_mapping_arr(output_format=self.model_type)

        filter_shape = (len(mapping), 3)

        filter = torch.zeros(
            filter_shape, dtype=self.dtype, device=self.device)
        for index, valid in enumerate(mapping > -1):
            if valid:
                filter[index] += 1

        # print("mapping:", get_named_joints(
        #     filter.detach().cpu().numpy(), ["shoulder-left", "hand-left", "elbow-left"]))
        return filter

    def forward(self, vpose_pose):

        bode_output = self.model(
            body_pose=self.body_pose + vpose_pose
        )

        # store model output for later renderer usage
        self.cur_out = bode_output

        joints = bode_output.joints
        # return a list with invalid joints set to zero
        filtered_joints = joints  # * self.filter.unsqueeze(0)
        return filtered_joints


def train_pose(
    model: SMPL,
    keypoints,
    keypoint_conf,
    camera: SimpleCamera,
    loss_layer=torch.nn.MSELoss(),
    learning_rate=1e-3,
    device=torch.device('cuda'),
    dtype=torch.float32,
    renderer: Renderer = None,
    optimizer=None,
    iterations=60
):
    # loss layers
    vposer = VPoserModel()
    vposer_layer = vposer.model
    vposer_params = vposer.get_vposer_latens()

    angle_prior_layer = SMPLifyAnglePrior()

    index = JOINT_NAMES.index("left_middle1")

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)
    keypoints_conf = torch.tensor(keypoint_conf).to(device)

    print("setup body pose...")

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device).to(device)

    if optimizer is None:
        parameters = [pose_layer.body_pose, vposer_params]
        optimizer = torch.optim.LBFGS(parameters, learning_rate)
        # optimizer = torch.optim.Adam(parameters, learning_rate)

    pbar = tqdm(total=iterations)

    def predict():
        body = vposer_layer()
        poZ = body.poZ_body

        # return joints based on current model state
        body_joints = pose_layer(body.pose_body)

        # compute homogeneous coordinates and project them to 2D space
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()

        # TODO: create custom cost function
        joint_loss = loss_layer(points, keypoints)

        # apply pose prior loss.
        prior_loss = poZ.pow(2).sum() * 2

        #angle_loss = angle_prior_layer(pose_layer.body_pose).sum() ** 2 * 0.05

        return joint_loss + prior_loss  # + angle_loss

    def optim_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = predict()

        if loss.requires_grad:
            loss.backward()
        return loss

    for t in range(iterations):
        optimizer.step(optim_closure)

        # LBFGS does not return the result, therefore we should rerun the model to get it
        with torch.no_grad():
            pred = predict()
            loss = optim_closure()

        # if t % 5 == 0:
        #     time.sleep(5)

        # compute loss
        cur_loss = loss.item()

        pbar.set_description("Error %f" % cur_loss)
        pbar.update(1)

        if renderer is not None:
            renderer.render_model(model, pose_layer.cur_out, keep_pose=True)
            R = camera.trans.numpy().squeeze()
            renderer.set_group_pose("body", R)

    pbar.close()
    print("Final result:", loss.item())
