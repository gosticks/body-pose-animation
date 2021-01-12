import json
import torch
import os
import numpy as np


class SMPLyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir="./samples",
            input_img_name_format="input_{id}_rendered.png",
            input_openpose_name_format="input_{id}_rendered.png",
            raw_img_format="frame-{id}.jpg"
    ):
        self.root_dir = root_dir
        self.input_img_name_format = input_img_name_format
        self.input_openpose_name_format = input_openpose_name_format
        self.raw_img_format = raw_img_format

    def __getitem__(self, index):
        name = str(index + 1).zfill(3) + ".json"
        path = os.path.join(
            self.root_dir, name)
        with open(path) as file:
            json_data = json.load(file)
            # FIXME: always take first person for now
            keypoints = json_data['people'][0]['pose_keypoints_2d']
        return self.transform(keypoints)
        # compute size of dataset based on items in folder
        # it is assumed that each "item" consists of 3 files

    def transform(self, data, origin_format="body_25", target_format="smpl"):
        # TODO: expand features as needed
        # based on mappings found here
        # https://github.com/ortegatron/playing_smplifyx/blob/master/smplifyx/utils.py
        map = np.array([
            # 8,  # hip - middle
            9,  # hip - left
            12,  # hip - right
            -1,  # body center (belly, not present in body_25)
            13,  # left knee
            10,  # right knee,
            1,  # chest
            14,  # left ankle
            11,  # right ankle
            1,  # chest again ? check this one out
            19,  # left toe
            22,  # right toe
            -1,  # neck (not present in body_25)
            -1,  # between torso and left shoulder
            -1,  # between torso and right shoulder
            0,  # head
            5,  # left shoulder
            2,  # right shoulder
            6,  # left elbow
            3,  # right elbow
            7,  # left hand
            4,  # right hand
            -1,  # left fingers
            -1  # right fingers

        ],
            dtype=np.int32)
        in_len = int(len(data) / 3)
        in_data = np.array(data).reshape((in_len, 3))
        out = np.zeros((len(map), 3), dtype=np.float32)

        for i in range(len(map)):
            m = map[i]
            if m == -1:
                continue
            # TODO: cleanup transform
            out[i][0] = (in_data[m][0] / 1920 * 2 - 1) * -1
            out[i][1] = (in_data[m][1] / 1080 * 2 - 1) * -1
        return out

    def __len__(self):
        # TODO: something like this could work for now we simply use one item
        # num_files = len(
        #     [name for name in os.listdir("./sample") if os.path.isfile(os.path.join("./sample", name))])
        # return num_files / 3
        return 1
