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
            start_index=1,
            # if config provided other values will be ignored
            config=None,
    ):
        self.root_dir = root_dir
        self.model_type = model_type
        self.size = size
        self.person_id = person_id
        self.sample_id_pad = sample_format.count('%')
        self.start_index = start_index

        if config is not None:
            self.root_dir = config['data']['path']
            self.size = config['data']['sampleCoords']
            self.person_id = config['data']['personId']
            self.model_type = config['smplType']
            self.img_format = config['data']['sampleImageFormat']
            self.sample_id_pad = config['data']['sampleImageFormat'].count('%')

        print(os.path.join(
            self.root_dir, self.get_item_name(0) + ".json"))

    def get_item_name(self, index):
        if self.sample_id_pad == 0:
            name = str(index)
        else:
            name = str(
                index + self.start_index).zfill(self.sample_id_pad)
        return name

    def __getitem__(self, index):
        name = self.get_item_name(index) + ".json"
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
        name = self.get_item_name(index) + ".png"
        path = os.path.join(self.root_dir, name)
        return path
