import yaml
import torch
import math

from model import *
from renderer import *
from dataset import *
from utils import get_smpl_joint

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


def align_torso():
    print("TODO: align torso here, use below code as inpsiration")

    # # rotate to match body initial position orientation
    # keypoints = keypoints * [1, -1, 0]

    # left_hip_2d = get_smpl_joint(keypoints, "hip-left")
    # right_hip_2d = get_smpl_joint(keypoints, "hip-right")

    # betas = torch.randn([1, 10], dtype=torch.float32)

    # # body_pose = torch.randn((1, 69), dtype=torch.float32) * 0.7
    # body_pose = torch.zeros((1, 69))

    # m.reset_params(
    #     betas=betas,
    #     body_pose=body_pose,
    # )

    # dtype = m.global_orient.dtype

    # global_orientation = m.global_orient.detach().cpu().numpy().squeeze()
    # joints_transl = m.transl.detach().cpu().numpy().squeeze()
    # print(global_orientation)

    # r.display_model(m, betas=betas,
    #                 keypoints=keypoints, body_pose=body_pose)

    # model_out = m()
    # joints = model_out.joints.detach().cpu().numpy().squeeze()
    # left_knee = get_smpl_joint(joints, "chest")
    # left_hip_joint = get_smpl_joint(
    #     joints, "hip-left")
    # right_hip_joint = get_smpl_joint(joints, "hip-right")

    # print("Error:", np.linalg.norm(right_hip_2d - right_hip_joint)
    #       ** 2 + np.linalg.norm(left_hip_2d - left_hip_joint) ** 2)

    # transl = torch.tensor(
    #     [left_hip_joint - left_knee], dtype=m.transl.dtype)

    # # print("left-hip", left_hip_joint)
    # # print("model trans:", m.transl)
    # # print("trans:", [left_hip_2d - left_hip_joint])

    # m.reset_params(
    #     transl=-transl
    # )
    # model_out = m()
    # # print(get_smpl_joint(joints, "hip-left"))

    # joints = model_out.joints.detach().cpu().numpy().squeeze()
    # global_orientation = m.global_orient.detach().cpu().numpy().squeeze()
    # left_hip_joint = get_smpl_joint(joints, "hip-left")
    # right_hip_joint = get_smpl_joint(joints, "hip-right")

    # print("left-hip", left_hip_joint)
    # print("model trans:", m.transl)

    # print("Error Left:", np.linalg.norm(left_hip_2d - left_hip_joint) ** 2)

    # print("Error:", np.linalg.norm(right_hip_2d - right_hip_joint)
    #       ** 2 + np.linalg.norm(left_hip_2d - left_hip_joint) ** 2)

    # # try moving left elbow
    # # body_pose.reshape(23, 3)[17][0] = math.pi / 4
    # # body_pose.reshape(23, 3)[17][1] = - math.pi / 2
    # # body_pose.reshape(23, 3)[17][2] = math.pi / 4

    # # # right knee
    # # body_pose.reshape(23, 3)[3][0] = math.pi / 4
    # # body_pose.reshape(23, 3)[3][1] = - math.pi / 2
    # # body_pose.reshape(23, 3)[3][2] = math.pi / 4

    # # # left knee
    # # body_pose.reshape(23, 3)[3][0] = math.pi / 4
    # # body_pose.reshape(23, 3)[3][1] = - math.pi / 2
    # # body_pose.reshape(23, 3)[3][2] = math.pi / 4

    # # body_pose = torch.from_numpy(keypoints.reshape((1, 69)))
    # # print(body_pose)
    # r.display_model(m, betas=betas,
    #                 keypoints=keypoints, body_pose=body_pose)


def main():
    print(ascii_logo)
    conf = load_config()
    print("config loaded")

    dataset = SMPLyDataset()

    l = SMPLyModel(conf['modelPath'])
    r = SMPLyRenderer()
    model = l.create_model()
    keypoints, conf = dataset[0]

    model_out = model()

    # print(keypoints, conf)


if __name__ == '__main__':
    main()
