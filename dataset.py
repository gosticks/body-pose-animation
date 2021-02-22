import json
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from utils.mapping import get_mapping_arr, apply_mapping, openpose_to_opengl_coords
import glob


class SMPLyDataset(Dataset):
    def __init__(
            self,
            root_dir="./samples",
            size: Tuple[int, int] = (1080, 1080),
            model_type="smplx",
            person_id=0,
            sample_format="%%%.json",
            image_format="%%%.png",
            start_index=1,
            sample_id_pad=None
    ):
        self.root_dir = root_dir
        self.model_type = model_type
        self.size = size
        self.person_id = person_id
        self.image_format = image_format
        self.sample_format = sample_format
        if sample_id_pad:
            self.sample_id_pad = sample_format.count('%')
        else:
            self.sample_id_pad = sample_id_pad
        self.start_index = start_index

    def from_config(config):
        """Create an instance of dataset based on the config

        Args:
            config ([type]): configuration object parsed from config.yaml

        Returns:
            [SMPLyDataset]: Dataset
        """
        return SMPLyDataset(
            root_dir=config['data']['rootDir'],
            size=config['data']['sampleCoords'],
            person_id=config['data']['personId'],
            model_type=config['smpl']['type'],
            image_format=config['data']['sampleImageFormat'],
            sample_format=config['data']['sampleNameFormat'],
            sample_id_pad=config['data']['sampleImageFormat'].count('%')
        )

    def get_image_id(self, index):
        img_id_pad = self.image_format.count("%")

        return str(index + self.start_index).zfill(img_id_pad)

    def get_item_name(self, index):
        if self.sample_id_pad == 0:
            name = str(index)
        else:
            name = str(
                index + self.start_index).zfill(self.sample_id_pad)
        return name

    def get_keypoint_name(self, index):
        id = self.get_item_name(index)
        return self.sample_format.replace("%" * len(id), id)

    def __getitem__(self, index):
        name = self.get_keypoint_name(index)
        path = os.path.join(
            self.root_dir, name)
        if os.path.exists(path):
            with open(path) as file:
                json_data = json.load(file)
                keypoints = json_data['people'][self.person_id]['pose_keypoints_2d']
            return self.transform(keypoints)
        else:
            print("[error]: no item at path ", path)
            return None

    def transform(self, data, origin_format="body_25"):
        """
            transform: transforms the order of an origin array to the target format
        """

        data = np.array(data).reshape((-1, 3))

        mapping = get_mapping_arr(origin_format, self.model_type)
        # remap data to match expacted target format
        remapped_data = apply_mapping(data, mapping)
        return openpose_to_opengl_coords(remapped_data, self.size[0], self.size[1])

    def __len__(self):
        sample_count = len(glob.glob1(self.root_dir, "*.json"))
        return sample_count

    def get_image_path(self, index):
        id = self.get_image_id(index)
        name = self.image_format.replace("%" * len(id), id)
        path = os.path.join(self.root_dir, name)

        return path
