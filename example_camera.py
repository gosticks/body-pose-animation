

from modules.camera import SimpleCamera
from modules.transform import Transform
from modules.pose import BodyPose
from renderer import Renderer
import torch
import torchgeometry as tgm
from model import *
# from renderer import *
from dataset import *
from utils.mapping import *
from utils.general import *

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
est_scale = estimate_scale(joints, keypoints)

# apply scaling to keypoints
keypoints = keypoints * est_scale

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
proj = SimpleCamera(dtype, device, 1)
optimizer = torch.optim.Adam(trans.parameters(), lr=learning_rate)
loss_layer = torch.nn.MSELoss()

for t in range(5000):
    points_h = tgm.convert_points_to_homogeneous(smpl_torso)
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
        R = trans.get_transform_mat(with_translate=True).numpy().squeeze()
        # update model rendering
        r.set_group_pose("body", R)


camera_transf = trans.get_transform_mat(with_translate=True).detach().cpu()
pose_layer = BodyPose(model, dtype=dtype, device=device)
camera = SimpleCamera(dtype, device, z_scale=1,
                      transform_mat=camera_transf)

op_kp = torch.Tensor(keypoints, device=device)
print(keypoints.shape)
pose_loss_layer = torch.nn.MSELoss()
pose_opt = torch.optim.Adam(pose_layer.parameters(), lr=1e-4)

print("starting training for pose...")
for t in range(2000):
    joints = pose_layer()
    points_h = tgm.convert_points_to_homogeneous(joints)
    points_2d = camera(points_h)

    # point wise differences
    diff = points_2d - op_kp

    loss = pose_loss_layer(op_kp, points_2d)

    if t % 100 == 99:
        print(t, loss.item())

    pose_opt.zero_grad()
    loss.backward()
    pose_opt.step()

    if t % 10 == 9:
        with torch.no_grad():
            # update model rendering
            r.render_model(model, pose_layer.cur_out)
