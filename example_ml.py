# library imports
import torch

# local imports
from train_pose import train_pose_with_conf
from modules.camera import SimpleCamera
from model import SMPLyModel
from utils.general import load_config, setup_training
from camera_estimation import TorchCameraEstimate
from dataset import SMPLyDataset

# load and select sample
config = load_config()
dataset = SMPLyDataset.from_config(config=config)
sample_index = 0

# prepare data and SMPL model
model = SMPLyModel.model_from_conf(config)
init_keypoints, init_joints, keypoints, conf, est_scale, r, img_path = setup_training(
    model=model,
    renderer=True,
    dataset=dataset,
    sample_index=sample_index
)

# configure PyTorch device and format
dtype = torch.float32
device = torch.device('cpu')


camera = TorchCameraEstimate(
    model,
    dataset=dataset,
    keypoints=keypoints,
    renderer=r,
    device=device,
    dtype=dtype,
    image_path=img_path,
    est_scale=est_scale
)

# render camera to the scene
camera.setup_visualization(r.init_keypoints, r.keypoints)

# run camera optimizer
cam, cam_trans, cam_int, cam_params = SimpleCamera.from_estimation_cam(
    camera,
    dtype=dtype,
    device=device,
)

# apply transform to scene
r.set_group_pose("body", cam_trans.cpu().numpy())


# train for pose
train_pose_with_conf(
    config=config,
    model=model,
    keypoints=keypoints,
    keypoint_conf=conf,
    camera=cam,
    renderer=r,
    device=device,
)
