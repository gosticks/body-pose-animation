# library imports
import numpy as np

# local imports
from renderer import DefaultRenderer
from train_pose import train_pose_with_conf
from modules.camera import SimpleCamera
from model import SMPLyModel
from utils.general import load_config, setup_training
from camera_estimation import TorchCameraEstimate
from dataset import SMPLyDataset

# this a simple pose playground with a async renderer for quick prototyping

# load and select sample
config = load_config()
model = SMPLyModel.model_from_conf(config)
# try changing angles
pose = model.body_pose

# left elbow
pose[:, 56] = -np.pi / 2
# right elbow
pose[:, 53] = np.pi / 2
# left knee
pose[:, 12] = np.pi / 2
# right knee
pose[:, 9] = np.pi / 2

model_out = model(
    pose=pose
)

r = DefaultRenderer()
r.setup(
    model,
    model_out=model_out
)

r.start()
