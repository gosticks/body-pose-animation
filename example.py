import yaml
import torch

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
    body_pose = torch.from_numpy(keypoints.reshape((1, 69)))
    print(body_pose)
    r.display_model(m, betas=betas, keypoints=keypoints, body_pose=body_pose)


if __name__ == '__main__':
    main()
