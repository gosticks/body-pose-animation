

import yaml
import torch
import math
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint
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


def get_torso_joints(model):
    joints = model_out.joints.detach().cpu().numpy().squeeze()
    cl_joint = get_named_joint(joints, "shoulder-left")
    cr_joint = get_named_joint(joints, "shoulder-right")
    torso_joints = torch.Tensor([cl_joint, cr_joint], device=device)
    return torso_joints


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

# SMPL joint positions (cl = chest left, cr = chest right)
cl_joint = get_named_joint(joints, "shoulder-left")
cr_joint = get_named_joint(joints, "shoulder-right")

# keypoint joint position
cl_point = get_named_joint(keypoints, "shoulder-left")
cr_point = get_named_joint(keypoints, "shoulder-right")

# create joints copy without points of interest
other_joints = np.array([joints[x]
                         for x in range(len(joints)) if x < 13 or x > 14])
print("removed other joints from list:", len(
    other_joints), other_joints.shape, len(joints))
scene = pyrender.Scene()

renderPoints(scene, other_joints)

# render shoulder joints
scene_joint = renderPoints(scene, [cl_joint, cr_joint],
                           radius=0.01, colors=[1.0, 0.0, 1.0, 1.0])

other_keypoints = np.array([keypoints[x]
                            for x in range(len(keypoints)) if x < 13 or x > 14])

# render openpose points
renderPoints(scene, other_keypoints,
             radius=0.005, colors=[0.0, 0.3, 0.0, 1.0])

renderPoints(scene, [cl_point, cr_point],
             radius=0.01, colors=[0.0, 0.7, 0.0, 1.0])


v = pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=True
                    )

# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------

smpl_torso = torch.Tensor([cl_joint, cr_joint], device=device)
keyp_torso = torch.Tensor([cl_point, cr_point], device=device)

orientation = torch.randn((3), device=device, dtype=dtype, requires_grad=True)
translation = torch.randn((3), device=device, dtype=dtype, requires_grad=True)

roll = torch.zeros(1,  device=device, dtype=dtype,  requires_grad=True)
yaw = torch.zeros(1,  device=device, dtype=dtype,  requires_grad=True)
pitch = torch.zeros(1,  device=device, dtype=dtype,  requires_grad=True)

tensor_0 = torch.zeros(1,  device=device, dtype=dtype)
tensor_1 = torch.ones(1,  device=device, dtype=dtype)

learning_rate = 0.1  # 1e-3
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

    # Compute cost function
    loss = torch.norm(pred - keyp_torso)
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
