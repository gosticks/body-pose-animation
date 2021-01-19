

from renderer import Renderer
import yaml
import torch
import time
import math
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint, get_named_joints, render_model, render_points
from collections import defaultdict

ascii_logo = """\
  /$$$$$$  /$$      /$$ /$$$$$$$  /$$   /$$     /$$
 /$$__  $$| $$$    /$$$| $$__  $$| $$  |  $$   /$$/
| $$  \__/| $$$$  /$$$$| $$  \ $$| $$   \  $$ /$$/
|  $$$$$$ | $$ $$/$$ $$| $$$$$$$/| $$    \  $$$$/
 \____  $$| $$  $$$| $$| $$____/ | $$     \  $$/
 /$$  \ $$| $$\  $ | $$| $$      | $$      | $$
|  $$$$$$/| $$ \/  | $$| $$      | $$$$$$$$| $$
 \______/ |__/     |__/|__/      |________/|__/

"""


dtype = torch.float
device = torch.device("cpu")


def load_config():
    with open('./config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


print(ascii_logo)
conf = load_config()
print("config loaded")
dataset = SMPLyDataset()


def estimate_depth(joints, keypoints, pairs=[
    ("shoulder-right", "hip-right"),
    ("shoulder-left", "hip-left")
], cam_fy=1):
    """estimate image depth based on the height changes due to perspective.
    This method only provides a rough estimate by computing shoulder to hip distances
    between SMPL joints and OpenPose keypoints.

    Args:
        joints ([type]): List of all SMPL joints
        keypoints ([type]): List of all OpenPose keypoints
        cam_fy (int, optional): Camera Y focal length. Defaults to 1.
    """

    # store distance vectors
    smpl_dists = []
    ops_dists = []

    for (j1, j2) in pairs:
        smpl_joints = get_named_joints(joints, [j1, j2])
        ops_keyp = get_named_joints(keypoints, [j1, j2])

        smpl_dists.append(smpl_joints[0] - smpl_joints[1])
        ops_dists.append(ops_keyp[0] - ops_keyp[1])

    smpl_height = np.linalg.norm(smpl_dists, axis=1).mean()
    ops_height = np.linalg.norm(ops_dists, axis=1).mean()

    return cam_fy * smpl_height / ops_height


# ------------------------------
# Load data
# ------------------------------
l = SMPLyModel(conf['modelPath'])
model = l.create_model()
keypoints, conf = dataset[0]
print("keypoints shape:", keypoints.shape)
# ---------------------------------
# Generate model and get joints
# ---------------------------------
model_out = model()
joints = model_out.joints.detach().cpu().numpy().squeeze()

# ---------------------------------
# Draw in the joints of interest
# ---------------------------------
cam_est_joints_names = ["hip-left", "hip-right",
                        "shoulder-left", "shoulder-right"]

est_depth = estimate_depth(joints, keypoints)

# apply depth to keypoints
keypoints[:, 2] = -est_depth

init_joints = get_named_joints(joints, cam_est_joints_names)
init_keypoints = get_named_joints(keypoints, cam_est_joints_names)


# setup renderer
r = Renderer()
r.render_model(model, model_out)
r.render_joints(joints)
r.render_keypoints(keypoints)

# render openpose torso markers
r.render_points(
    init_keypoints,
    radius=0.01,
    color=[1.0, 0.0, 1.0, 1.0], name="ops_torso", group_name="keypoints")

r.render_points(
    init_joints,
    radius=0.01,
    color=[0.0, 0.7, 0.0, 1.0], name="body_torso", group_name="body")


# start renderer
r.start()

# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------

smpl_torso = torch.Tensor(init_joints, device=device)
keyp_torso = torch.Tensor(init_keypoints, device=device)

translation = torch.zeros((3), device=device, dtype=dtype, requires_grad=True)

roll = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)
yaw = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)
pitch = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)

tensor_0 = torch.zeros(1,  device=device, dtype=dtype)
tensor_1 = torch.ones(1,  device=device, dtype=dtype)

learning_rate = 1e-3
for t in range(200000):
    RX = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    pred = smpl_torso @ R + translation

    # apply 2d projection here

    # point wise differences
    diff = pred - keyp_torso

    # Compute cost function
    loss = torch.norm(diff[:, :, :2])
    if t % 100 == 99:
        print(t, loss)

    loss.backward()

    with torch.no_grad():
        # update model rendering
        r.set_group_transform("body", R.numpy(), translation.numpy())

        translation -= learning_rate * translation.grad
        roll -= learning_rate * roll.grad
        yaw -= learning_rate * yaw.grad
        pitch -= learning_rate * pitch.grad

        translation.grad = None
        roll.grad = None
        yaw.grad = None
        pitch.grad = None

print("roll, yaw, pitch:", roll, yaw, pitch)
print("transl:", translation)

# -----------------------------
# Render the points
# -----------------------------
# v = pyrender.Viewer(scene,
#                     use_raymond_lighting=True,
#                     # show_world_axis=True
#                     run_in_thread=True
#                     )
