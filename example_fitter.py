

import yaml
import torch
import math

from model import *
# from renderer import *
from dataset import *
from utils import get_named_joint

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
mapping = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

arr = np.ones(45) * -1

for i, v in enumerate(mapping):
    arr[v] = i
    print(v, i)

for v in arr:
    print(
        int(v), ","
    )
print(arr)

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
cl_joint = get_named_joint(joints, "elbow-left")
cr_joint = get_named_joint(joints, "hip-left")

# keypoint joint position
cl_point = get_named_joint(keypoints, "shoulder-left", type="body_25")
cr_point = get_named_joint(keypoints, "shoulder-right", type="body_25")

# create joints copy without points of interest
other_joints = np.array([joints[x]
                         for x in range(len(joints)) if x < 13 or x > 14])
print("removed other joints from list:", len(
    other_joints), other_joints.shape, len(joints))
scene = pyrender.Scene()

renderPoints(scene, other_joints)

renderPoints(scene, [cl_joint, cr_joint],
             radius=0.01, colors=[1.0, 0.0, 1.0, 1.0])

v = pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=False
                    )


# -------------------------------------
# Optimize for translation and rotation
# -------------------------------------

# -----------------------------
# Render the points
# -----------------------------
v = pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    # show_world_axis=True
                    run_in_thread=True
                    )
