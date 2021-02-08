# library imports
from utils.render import make_video
import torch
import matplotlib.pyplot as plt

# local imports
from train_pose import train_pose_with_conf
from modules.camera import SimpleCamera
from model import SMPLyModel
from utils.general import getfilename_from_conf, load_config, setup_training
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
    sample_index=sample_index,
    offscreen=True
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


# train for pose
result, best, train_loss, step_imgs = train_pose_with_conf(
    config=config,
    model=model,
    keypoints=keypoints,
    keypoint_conf=conf,
    camera=camera,
    renderer=r,
    device=device,
)


make_video(step_imgs, "test.avi")

# color = r.get_snapshot()
# plt.imshow(color)
# plt.show()

# fig, ax = plt.subplots()
# name = getfilename_from_conf(config=config, index=sample_index)
# ax.plot(train_loss[1::], label='sgd')
# ax.set(xlabel="Training iteration", ylabel="Loss", title='Training loss')
# fig.savefig("results/" + name + ".png")
# ax.legend()
# plt.show()
