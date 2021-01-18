

import yaml
import torch
import math
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint, get_named_joints
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


def renderPoints(scene, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0], name=None):
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = colors
    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:, :3, 3] = points
    pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # return the render scene node
    return scene.add(pcl, name=name)


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

# FIND OPENPOSE TO SMPL MAPPINGS
# mapping = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
#            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# arr = np.ones(45) * -1

# for i, v in enumerate(mapping):
#     arr[v] = i
#     print(v, i)

# for v in arr:
#     print(
#         int(v), ","
#     )
# print(arr)

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

init_joints = get_named_joints(joints, cam_est_joints_names)
init_keypoints = get_named_joints(keypoints, cam_est_joints_names)

# compute height of both 2d and 3d points
joint_torso_height = np.array([
    init_joints[3] - init_joints[1],
    init_joints[2] - init_joints[0]
]).squeeze()

keypoint_torso_height = np.array([
    init_keypoints[3] - init_keypoints[1],
    init_keypoints[2] - init_keypoints[0]
]).squeeze()


jt_lengths = np.linalg.norm(joint_torso_height, axis=1)
kp_lengths = np.linalg.norm(keypoint_torso_height, axis=1)
print(jt_lengths, kp_lengths)

jp_height = jt_lengths.mean()
kp_height = kp_lengths.mean()

scene = pyrender.Scene()

renderPoints(scene, joints)

# render shoulder joints
scene_joint = renderPoints(scene, init_joints,
                           radius=0.01, colors=[1.0, 0.0, 1.0, 1.0])

# render openpose points
renderPoints(scene, keypoints,
             radius=0.005, colors=[0.0, 0.3, 0.0, 1.0])

renderPoints(scene, init_keypoints,
             radius=0.01, colors=[0.0, 0.7, 0.0, 1.0])


v = pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=True
                    )

# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------

smpl_torso = torch.Tensor(init_joints, device=device)
keyp_torso = torch.Tensor(init_keypoints, device=device)

orientation = torch.randn((3), device=device, dtype=dtype, requires_grad=True)
translation = torch.randn((3), device=device, dtype=dtype, requires_grad=True)

roll = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)
yaw = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)
pitch = torch.randn(1,  device=device, dtype=dtype,  requires_grad=True)

tensor_0 = torch.zeros(1,  device=device, dtype=dtype)
tensor_1 = torch.ones(1,  device=device, dtype=dtype)

learning_rate = 1e-3
for t in range(20000):
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
    diff = pred - keyp_torso
    # Compute cost function
    loss = torch.norm(diff[:, :, :2])
    if t % 100 == 99:
        print(t, loss)

    loss.backward()

    with torch.no_grad():
        pose = np.eye(4)
        pose[:3, :3] = R.numpy()
        pose[:3, 3] = translation.numpy()

        v.render_lock.acquire()
        scene.set_pose(scene_joint, pose)
        v.render_lock.release()

        # orientation -= learning_rate * orientation.grad
        translation -= learning_rate * translation.grad
        roll -= learning_rate * roll.grad
        yaw -= learning_rate * yaw.grad
        pitch -= learning_rate * pitch.grad

        translation.grad = None
        roll.grad = None
        yaw.grad = None
        pitch.grad = None
        # orientation.grad = None
        # model.global_orient -= learning_rate * model.global_orient.grad
        # model.transl -= learning_rate * model.transl
        # model.global_orient.grad = None
        # model.transl.grad = None

        # model.reset_params(**defaultdict(
        #     transl=model.transl,
        #     global_orient=model.global_orient
        # ))

print("roll, yaw, pitch:", roll, yaw, pitch)
print("transl:", translation)

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

rot = R.detach().cpu().numpy()

torso = smpl_torso.detach().cpu().numpy(
).squeeze()
transl = translation.detach().cpu().numpy().squeeze()

new_points = torso.dot(rot) + transl
renderPoints(scene, [new_points[0]],
             radius=0.05, colors=[0.7, 0.3, 0.0, 1.0])
renderPoints(scene, [new_points[1]],
             radius=0.05, colors=[0.0, 0.3, 0.7, 1.0])
# -----------------------------
# Render the points
# -----------------------------
# v = pyrender.Viewer(scene,
#                     use_raymond_lighting=True,
#                     # show_world_axis=True
#                     run_in_thread=True
#                     )
