import json
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from utils.mapping import get_mapping_arr, apply_mapping, openpose_to_opengl_coords


class SMPLyDataset(Dataset):
    def __init__(
            self,
            root_dir="./samples",
            size: Tuple[int, int] = (1080, 1080),
            model_type="smplx"
    ):
        self.root_dir = root_dir
        self.model_type = model_type
        self.size = size

    def __getitem__(self, index):
        name = str(index) + ".json"
        path = os.path.join(
            self.root_dir, name)
        if os.path.exists(path):
            with open(path) as file:
                json_data = json.load(file)
                # FIXME: always take first person for now
                keypoints = json_data['people'][0]['pose_keypoints_2d']
            return self.transform(keypoints)
        # compute size of dataset based on items in folder
        # it is assumed that each "item" consists of 3 files

    def transform(self, data, origin_format="body_25"):
        """ 
            transform: transforms the order of an origin array to the target format
        """

        data = np.array(data).reshape((-1, 3))

        mapping = get_mapping_arr(origin_format, self.model_type)
        # remap data to match expacted target format
        remapped_data = apply_mapping(data, mapping)
        # TODO: pass image resolution here
        return openpose_to_opengl_coords(remapped_data, self.size[0], self.size[1])

    def __len__(self):
        # TODO: something like this could work for now we simply use one item
        # num_files = len(
        #     [name for name in os.listdir("./sample") if os.path.isfile(os.path.join("./sample", name))])
        # return num_files / 3
        return 3

    def get_image_path(self, index):
        name = str(index) + ".png"
        path = os.path.join(self.root_dir, name)
        return path
