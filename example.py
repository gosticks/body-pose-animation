import yaml
import torch
import math

from model import *
from renderer import *
from dataset import *

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


def load_config():
    with open('./config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def main():
    print(ascii_logo)
    conf = load_config()
    print("config loaded")

    dataset = SMPLyDataset()

    l = SMPLyModel(conf['modelPath'])
    r = SMPLyRenderer()
    m = l.create_model()

    keypoints = dataset[0]
    print(keypoints)
    betas = torch.randn([1, 10], dtype=torch.float32)

    # torch.randn((1, 69), dtype=torch.float32) * 0.7
    body_pose = torch.zeros((1, 69))

    # try moving left elbow
    body_pose.reshape(23, 3)[17][0] = math.pi / 4
    body_pose.reshape(23, 3)[17][1] = - math.pi / 2
    body_pose.reshape(23, 3)[17][2] = math.pi / 4

    # right knee
    body_pose.reshape(23, 3)[3][0] = math.pi / 4
    body_pose.reshape(23, 3)[3][1] = - math.pi / 2
    body_pose.reshape(23, 3)[3][2] = math.pi / 4

    # left knee
    body_pose.reshape(23, 3)[3][0] = math.pi / 4
    body_pose.reshape(23, 3)[3][1] = - math.pi / 2
    body_pose.reshape(23, 3)[3][2] = math.pi / 4

    # body_pose = torch.from_numpy(keypoints.reshape((1, 69)))
    print(body_pose)
    r.display_model(m, betas=betas, keypoints=keypoints, body_pose=body_pose)


if __name__ == '__main__':
    main()
