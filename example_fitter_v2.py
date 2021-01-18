

from camera import Camera
import yaml
import torch
import math
import time
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint, get_named_joints
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn


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


def render_model(
    scene,
    model_out,
    color=[0.3, 0.3, 0.3, 0.8],
    name=None,
    replace=False,
):
    vertices = model_out.vertices.detach().cpu().numpy().squeeze()

    # set vertex colors, maybe use this to highlight accuracies
    vertex_colors = np.ones([vertices.shape[0], 4]) * color

    # triangulate vertex mesh
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    if name is not None and replace:
        for node in scene.get_nodes(name=name):
            scene.remove_node(node)

    return scene.add(mesh, name=name)


def render_points(scene, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0], name=None):
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


# create joints copy without points of interest
scene = pyrender.Scene()

# render smpl joints
render_points(scene, joints)

# render openpose points
render_points(scene, keypoints,
              radius=0.005, colors=[0.0, 0.3, 0.0, 1.0])


# --------------------------------------
# Create camera matrix
# --------------------------------------
cam_width = 1920
cam_height = 1080
# focal_length for x and y
cam_focal_x = 5000
cam_focal_y = 5000

est_d = cam_focal_y * (jp_height / kp_height)

cam_pose = np.eye(4)
cam_pose[:3, 3] = np.array([0, 0, -est_d])
cam_pose[0, 0] *= -1.0

# add camera
camera = pyrender.camera.IntrinsicsCamera(
    fx=cam_focal_x,
    fy=cam_focal_y,
    cx=1000,
    cy=1000
)

scene.add(camera, pose=cam_pose)

render_model(scene, model_out, name="body_model", replace=True)

v = pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=True
                    )

# -------------------------------------
# Do basic camera position estimations
# -------------------------------------

smpl_torso = torch.Tensor(init_joints, device=device)
keyp_torso = torch.Tensor(init_keypoints, device=device)


# try guessing z distance based on hight differences


# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------
learning_rate = 1e-3


rotation = torch.eye(
    3, dtype=dtype).unsqueeze(dim=0).repeat(smpl_torso.shape[0], 1, 1)

translation = torch.zeros(
    (smpl_torso.shape[0], 3),
    device=device,
    dtype=dtype)
translation = nn.Parameter(translation, requires_grad=True)

optimizer = torch.optim.Adam([translation, model.global_orient], learning_rate)

center = torch.zeros(3, device=device, dtype=dtype)
camera = Camera(
    rotation=rotation,
    translation=translation,
    fx=cam_focal_x,
    fy=cam_focal_y,
    dtype=dtype,
    device=device,
    batch_size=smpl_torso.shape[0],
    center=center
)

print(camera(smpl_torso))

for t in range(2000):

    def optimizer_closure():
        # reset gradient for all parameters
        optimizer.zero_grad()
        # Compute cost function (LSE)
        loss = torch.sum(torch.dist(pred - homog_keyp) ** 2)
        if t % 100 == 99:
            print(t, loss)

        with torch.no_grad():
            pose = np.eye(4)
            pose[:3, 3] = translation.numpy()

            v.render_lock.acquire()

            model.reset_params(**defaultdict(
                transl=translation,
                global_orient=model.global_orient,
            ))

            model_out = model()
            render_model(scene, model_out, name="body_model", replace=True)
            v.render_lock.release()
        loss.backward()
        return loss

    optimizer.step(optimizer_closure)


# print("roll, yaw, pitch:", roll, yaw, pitch)
# print("transl:", translation)

# RX = torch.stack([
#     torch.stack([tensor_1, tensor_0, tensor_0]),
#     torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
#     torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

# RY = torch.stack([
#     torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
#     torch.stack([tensor_0, tensor_1, tensor_0]),
#     torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

# RZ = torch.stack([
#     torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
#     torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
#     torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

# R = torch.mm(RZ, RY)
# R = torch.mm(R, RX)

# rot = R.detach().cpu().numpy()

# torso = smpl_torso.detach().cpu().numpy(
# ).squeeze()
# transl = translation.detach().cpu().numpy().squeeze()

# new_points = torso.dot(rot) + transl
# render_points(scene, [new_points[0]],
#               radius=0.05, colors=[0.7, 0.3, 0.0, 1.0])
# render_points(scene, [new_points[1]],
#               radius=0.05, colors=[0.0, 0.3, 0.7, 1.0])
# -----------------------------
# Render the points
# -----------------------------
# v = pyrender.Viewer(scene,
#                     use_raymond_lighting=True,
#                     # show_world_axis=True
#                     run_in_thread=True
#                     )
