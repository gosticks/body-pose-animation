

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
from camera_estimation import TorchCameraEstimate


# mapping = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#            8, 1, 4, 7, 56, 57, 58, 59]  # 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]


# arr = np.ones(127) * -1  # arr = np.ones(45) * -1


# for i, v in enumerate(mapping):  # for i, v in enumerate(mapping):
#     arr[v] = i  # arr[v] = i
#     print(v, i)  # print(v, i)


# for v in arr:  # for v in arr:
#     print(  # print(
#         int(v), ","  # int(v), ","
#     )  # )
# print(arr)  # print(arr)


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


sample_index = 2

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


r = Renderer()

# integrating Camera Estimation

init_joints = get_torso(joints)
init_keypoints = get_torso(keypoints)

print("image path:", img_path)
# setup renderer

r.render_model(model, model_out)
# r.render_joints(joints)
# r.render_keypoints(keypoints)
r.render_image_from_path(img_path, est_scale)

# render openpose torso markers
render_keypoints = r.render_points(
    keypoints,
    radius=0.005,
    color=[1.0, 1.0, 1.0, 1.0])

render_keypoints = r.render_keypoints(
    init_keypoints,
    radius=0.01,
    color=[1.0, 0.0, 1.0, 1.0])

render_points = r.render_points(
    init_joints,
    radius=0.01,
    color=[0.0, 0.1, 0.0, 1.0], name="torso", group_name="body")


camera = TorchCameraEstimate(
    model,
    dataset=dataset,
    keypoints=keypoints,
    renderer=r,
    device=torch.device('cpu'),
    dtype=torch.float32,
    image_path=img_path,
    est_scale=est_scale
)
pose, transform, cam_trans = camera.estimate_camera_pos()

camera.setup_visualization(render_points, render_keypoints)


# start renderer
# r.start()

dtype = torch.float
device = torch.device('cpu')
camera_transformation = transform.clone().detach().to(device=device, dtype=dtype)
camera_int = pose.clone().detach().to(device=device, dtype=dtype)
camera_params = cam_trans.clone().detach().to(device=device, dtype=dtype)

camera = SimpleCamera(dtype, device,
                      transform_mat=camera_transformation,
                      #   camera_intrinsics=camera_int, camera_trans_rot=camera_params
                      )

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
    iterations=60
)
