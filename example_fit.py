

from modules.camera import SimpleCamera
from modules.transform import Transform
from modules.pose import BodyPose, train_pose
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
print(ascii_logo)
conf = load_config()
print("config loaded")
dataset = SMPLyDataset()


# ------------------------------
# Load data
# ------------------------------
l = SMPLyModel(conf['modelPath'])
model = l.create_model()
keypoints, conf = dataset[2]
# ---------------------------------
# Generate model and get joints
# ---------------------------------
model_out = model()
joints = model_out.joints.detach().cpu().numpy().squeeze()

# ---------------------------------
# Draw in the joints of interest
# ---------------------------------
est_scale = estimate_scale(joints, keypoints)

# apply scaling to keypoints
keypoints = keypoints * est_scale

init_joints = get_torso(joints)
init_keypoints = get_torso(keypoints)


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
init_keypoints = get_torso(keypoints)

# start renderer
r.start()


dtype = torch.float
device = torch.device('cpu')
# torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# camera_transformation = torch.tensor([
#     [0.929741,   -0.01139284,  0.36803687, 0.68193704],
#     [0.01440641,  0.999881,   -0.00544171, 0.35154277],
#     [-0.36793125, 0.01036147,  0.9297949,  0.52250534],
#     [0, 0, 0, 1]
# ]).to(device=device, dtype=dtype)
# camera_transformation = torch.tensor(
#     [[0.9993728,  -0.00577453,  0.03493736,  0.9268496],
#      [0.00514091,  0.9998211,   0.01819922, -0.07861858],
#         [-0.0350362,  -0.0180082,   0.99922377,  0.00451744],
#         [0,          0,          0,          1]]
# ).to(device=device, dtype=dtype)

# camera_transformation = torch.tensor(
#     [[ 4.9928,  0.0169,  0.5675,  0.3011],
#         [-0.0289,  4.9951,  0.5460,  0.1138],
#         [-0.0135, -0.0093,  0.9999,  5.4520],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]]
# ).to(device=device, dtype=dtype)
camera_transformation = torch.from_numpy(np.eye(4)).to(device=device, dtype=dtype)

camera = SimpleCamera(dtype, device, z_scale=1,
                      transform_mat=camera_transformation)

r.set_group_pose("body", camera_transformation.detach().cpu().numpy())

print("using device", device)


train_pose(
    model,
    keypoints=keypoints,
    keypoint_conf=conf,
    # TODO: use camera_estimation camera here
    camera=camera,
    renderer=r,
    device=device,
    iterations=25
)
