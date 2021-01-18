

import yaml
import torch
import math
import time
from math import cos, sin

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint
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
    model,
    color=[0.3, 0.3, 0.3, 0.8],
    name=None,
    replace=False,
):
    model_out = model()
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

render_points(scene, other_joints)

# render shoulder joints
scene_joint_left = render_points(scene, [cl_joint],
                                 radius=0.01, colors=[1.0, 0.0, 1.0, 1.0])

scene_joint_right = render_points(scene, [cr_joint],
                                  radius=0.01, colors=[1.0, 1.0, 0.0, 1.0])

other_keypoints = np.array([keypoints[x]
                            for x in range(len(keypoints)) if x < 13 or x > 14])

scene_model = render_model(scene, model)

# render openpose points
render_points(scene, other_keypoints,
              radius=0.005, colors=[0.0, 0.3, 0.0, 1.0])

render_points(scene, [cl_point, cr_point],
              radius=0.01, colors=[0.0, 0.7, 0.0, 1.0])


# --------------------------------------
# Create camera matrix
# --------------------------------------
cam_width = 1920
cam_height = 1080
# focal_length for x and y
cam_focal_x = 1000
cam_focal_y = 1000

f = 4.25
cam_focal_x = f * cam_width / 5.76
cam_focal_y = f * cam_height / 4.29
cx = cam_width/2.
cy = cam_height/2.


cam_pose = np.eye(4)
cam_pose[:3, 3] = [1, 1, 0]
cam_pose[0, 0] *= -1.0
# cam_pose[2, 2] *= -1.0

# add camera
camera = pyrender.camera.IntrinsicsCamera(
    fx=cam_focal_x,
    fy=cam_focal_y,
    cx=cx,
    cy=cy
)

scene.add(camera, pose=cam_pose)

v = pyrender.Viewer(scene,
                    viewport_size=(1920, 1080),

                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=True
                    )

# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------
learning_rate = 1e-3


smpl_torso = torch.Tensor([cl_joint, cr_joint], device=device)
keyp_torso = torch.Tensor([cl_point, cr_point], device=device)
# homog_coord_keyp = torch.ones(list(keyp_torso.shape)[:-1] + [1],
#                               dtype=dtype,
#                               device=device)
# homog_keyp = torch.cat([keyp_torso, homog_coord_keyp], dim=-1)

orientation = torch.eye(
    3,
    device=device,
    dtype=dtype
).unsqueeze(dim=0).repeat(2, 1, 1)
orientation = nn.Parameter(orientation, requires_grad=True)

translation = torch.zeros(
    [2, 3],
    device=device,
    dtype=dtype)
translation = nn.Parameter(translation, requires_grad=True)

optimizer = torch.optim.Adam([translation, orientation], learning_rate)

center = torch.from_numpy(np.array([[cam_width / 2, cam_height / 2]]))


for t in range(2000):

    def optimizer_closure():
        # reset gradient for all parameters
        optimizer.zero_grad()

        with torch.no_grad():
            camera_mat = torch.zeros([2, 2, 2],
                                     dtype=dtype, device=device)
            camera_mat[:, 0, 0] = cam_focal_x
            camera_mat[:, 1, 1] = cam_focal_x

        transform = torch.cat([F.pad(orientation, (0, 0, 0, 1), "constant", value=0),
                               F.pad(translation.unsqueeze(dim=-1), (0, 0, 0, 1), "constant", value=1)], dim=2)
        # orientation
        # transform[0:3, 3] = translation
        # transform = transform.unsqueeze(0)

        # convert points to homogenious coordinates
        homog_coord = torch.ones(list(smpl_torso.shape)[:-1] + [1],
                                 dtype=dtype,
                                 device=device)

        homog_torso = torch.cat([smpl_torso, homog_coord], dim=-1)

        proj = torch.einsum('bki,bji->bjk', [transform, homog_torso])
        print("proj:", proj)
        img_points = torch.div(proj[:, :, :2],
                               proj[:, :, 2].unsqueeze(dim=-1))

        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + center.unsqueeze(dim=1)

        print("img_points:", img_points)

        # transform coordinates to image space coordinates

        # Compute cost function (LSE)
        loss = torch.sum(torch.dist(pred - homog_keyp) ** 2)
        if t % 100 == 99:
            print(t, loss)

        with torch.no_grad():
            v.render_lock.acquire()

            scene.set_pose(scene_joint_left, transform[0].numpy())
            scene.set_pose(scene_joint_right, transform[1].numpy())
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
