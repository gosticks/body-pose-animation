

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


sample_index = 1

sample_transforms = [
    [
        [0.929741,   -0.01139284,  0.36803687, 0.68193704],
        [0.01440641,  0.999881,   -0.00544171, 0.35154277],
        [-0.36793125, 0.01036147,  0.9297949,  0.52250534],
        [0, 0, 0, 1]
    ],
    [
        [9.9901e-01, -3.7266e-02, -2.4385e-02,  7.6932e-01],
        [3.5270e-02,  9.9635e-01, -7.7715e-02,  3.0069e-01],
        [2.7193e-02,  7.6778e-02,  9.9668e-01, -7.6563e-04],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
    ],
    [
        [9.99947985e-01, - 7.05885983e-03, -7.36209961e-03,  8.18256989e-01],
        [7.58265353e-03,  9.97249329e-01,  7.37311259e-02, - 6.41522022e-02],
        [6.82139121e-03, - 7.37831150e-02,  9.97250982e-01,  6.04774204e-04],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ],
    [
        [4.9928,  0.0169,  0.5675,  0.3011],
        [-0.0289,  4.9951,  0.5460,  0.1138],
        [-0.0135, -0.0093,  0.9999,  5.4520],
        [0.0000,  0.0000,  0.0000,  1.0000]
    ]
]


# ------------------------------
# Load data
# ------------------------------
l = SMPLyModel(conf['modelPath'])
model = l.create_model()
keypoints, conf = dataset[sample_index]
img_path = "./samples/" + str(sample_index + 1).zfill(3) + ".png"
print(img_path)
# ---------------------------------
# Generate model and get joints
# ---------------------------------
model_out = model()
joints = model_out.joints.detach().cpu().numpy().squeeze()

# ---------------------------------
# Draw in the joints of interest
# ---------------------------------
est_scale = estimate_scale(joints, keypoints)
print("ESTIMATED SCALE:", est_scale)

# apply scaling to keypoints
keypoints = keypoints * est_scale

init_joints = get_torso(joints)
init_keypoints = get_torso(keypoints)

# setup renderer
r = Renderer()
r.render_model(model, model_out)
# r.render_joints(joints)
# r.render_keypoints(keypoints)
# r.render_image_from_path(img_path)

# render openpose torso markers
r.render_keypoints(
    init_keypoints,
    radius=0.01,
    color=[1.0, 0.0, 1.0, 1.0])

r.render_points(
    init_joints,
    radius=0.01,
    color=[0.0, 0.1, 0.0, 1.0], name="torso", group_name="body")

# start renderer
r.start()


dtype = torch.float
device = torch.device('cpu')
camera_transformation = torch.tensor(
    sample_transforms[sample_index]).to(device=device, dtype=dtype)

camera = SimpleCamera(dtype, device, z_scale=1,
                      transform_mat=camera_transformation)

r.set_group_pose("body", camera_transformation.detach().cpu().numpy())

print("using device", device)


train_pose(
    model,
    learning_rate=1e-2,
    keypoints=keypoints,
    keypoint_conf=conf,
    # TODO: use camera_estimation camera here
    camera=camera,
    renderer=r,
    device=device,
    iterations=30
)
