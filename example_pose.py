# library imports
import numpy as np
import time
# local imports
from renderer import DefaultRenderer
from model import SMPLyModel
from utils.general import load_config

# this a simple pose playground with a async renderer for quick prototyping

# load and select sample
config = load_config()
model = SMPLyModel.model_from_conf(config)
# try changing angles
pose = model.body_pose

initial_pose = model.body_pose.detach().clone()

# left elbow
# pose[:, 56] = -np.pi / 2
# # right elbow
# pose[:, 53] = np.pi / 2
# # left knee
# pose[:, 12] = np.pi / 2
# # right knee
# pose[:, 9] = np.pi / 2

# right knee

# back x axis rotation
#pose[:, 24] = np.pi / 2
# back y axis rotation
#pose[:, 25] = np.pi / 2

# back left rotation
#pose[:, 26] = np.pi / 2

# back left rotation
# pose[:, 1] = np.pi / 2

pose[:, 37] = -np.pi / 2
pose[:, 40] = np.pi / 2


model_out = model(
    pose=pose
)

r = DefaultRenderer()
r.setup(
    model,
    model_out=model_out
)

r.start()

# for angle_idx in range(len(pose[0])):

#     print("angle:", str(angle_idx))

#     pose_copy = initial_pose.detach().clone()
#     pose_copy[:, angle_idx] = np.pi / 2
#     model_out = model(
#         body_pose=pose_copy
#     )

#     r.render_model(model, model_out)

#     time.sleep(1)
