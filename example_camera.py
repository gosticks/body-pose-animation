# library imports
import torch

# local imports
from model import SMPLyModel
from utils.general import load_config, setup_training
from camera_estimation import TorchCameraEstimate
from renderer import Renderer
from dataset import SMPLyDataset

dtype = torch.float32
device = torch.device('cpu')
sample_index = 0

config = load_config()
dataset = SMPLyDataset.from_config(config=config)
model = SMPLyModel.model_from_conf(config)
init_keypoints, init_joints, keypoints, conf, est_scale, r, img_path = setup_training(
    model=model,
    renderer=True,
    dataset=dataset,
    sample_index=sample_index
)


camera = TorchCameraEstimate(
    model,
    keypoints=keypoints,
    renderer=Renderer(),
    device=device,
    dtype=dtype,
    image_path=dataset.get_item_name(sample_index)
)

pose, transform, cam_tansform = camera.estimate_camera_pos()
print("Pose matrix: \n", pose)

print("Transform matrix: \n", transform)
