

from modules.camera import CameraProjSimple
from modules.transform import Transform
from renderer import Renderer
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import tan, radians
from model import *
import time
# from renderer import *
from dataset import *
from utils import get_named_joints, estimate_depth

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
# torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')


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

keypoints[:, 2] = 0
init_keypoints = get_named_joints(keypoints, cam_est_joints_names)

# start renderer
r.start()

# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------

smpl_torso = torch.Tensor(init_joints, device=device)
keyp_torso = torch.Tensor(init_keypoints, device=device)


learning_rate = 1e-3
trans = Transform(dtype, device)
proj = CameraProjSimple(dtype, device, -est_depth)
optimizer = torch.optim.Adam(trans.parameters(), lr=learning_rate)
loss_layer = torch.nn.MSELoss()

for t in range(50000):
    homog_coord = torch.ones(list(smpl_torso.shape)[:-1] + [1],
                                 dtype=smpl_torso.dtype,
                                 device=device)
    # Convert the points to homogeneous coordinates
    points_h = torch.cat([smpl_torso, homog_coord], dim=-1)

    points = trans(points_h)
    points_2d = proj(points)

    # point wise differences
    diff = points_2d - keyp_torso

    # Compute cost function
    # loss = torch.norm(diff)
    loss = loss_layer(keyp_torso, points_2d)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        R = trans.get_transform_mat().numpy()
        translation = trans.translation.detach().numpy()
        # update model rendering
        r.set_homog_group_transform("body", R, translation)


# for t in range(50000):
#     print("do cost evaluation here")
