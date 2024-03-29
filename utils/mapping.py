
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
from trimesh.triangles import normals

openpose_to_smpl = np.array([
    8,  # hip - middle / pelvis
    12,  # hip - right
    9,  # hip - left
    -1,  # body center (belly, not present in body_25)
    13,  # left knee
    10,  # right knee,
        -1,
    14,  # left ankle
    11,  # right ankle
        -1,
        -1,
        -1,
    1,  # chest
        -1,
        -1,
        -1,
    5,  # left shoulder
    2,  # right shoulder
    6,  # left elbow
    3,  # right elbow
    7,  # left hand
    4,  # right hand
        -1,
        -1,
    0,  # head
    15,
    16,
    17,
    18,
    19,  # left toe
    20,
    21,
    22,  # right toe
    23,
    24,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
])

openpose_to_smplx = np.array([8,
                              12,
                              9,
                              -1,
                              13,
                              10,
                              -1,
                              14,
                              11,
                              -1,
                              -1,
                              -1,
                              1,
                              -1,
                              -1,
                              -1,
                              5,
                              2,
                              6,
                              3,
                              7,
                              4,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              0,
                              15,
                              16,
                              17,
                              18,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1,
                              -1])


def get_mapping_arr(
    input_format: str = "body_25",
    output_format: str = "smpl",
) -> list:

    # TODO: expand features as needed
    # based on mappings found here
    # https://github.com/ortegatron/playing_smplifyx/blob/master/smplifyx/utils.py
    if output_format == "smpl":
        return openpose_to_smpl
    if output_format == "smplx":
        # create a list of length 127 and pad all values beyond 47 with -1 since we do not perform face and finger detection
        # return np.pad(
        #     openpose_to_smpl,
        #     (0, 127-len(openpose_to_smpl)),
        #     constant_values=(0, -1))
        return openpose_to_smplx


joint_names_body_25 = {
    "hip-left": 9,
    "hip-right": 12,
    "belly": 8,
    "knee-left": 10,
    "knee-right": 13,
    "ankle-left": 11,
    "ankle-right": 14,
    "toes-left": 22,
    "toes-right": 19,
    "neck": 1,
    "head": 0,
    "shoulder-left": 2,
    "shoulder-right": 5,
    "elbow-left": 3,
    "elbow-right": 6,
    "hand-left": 4,
    "hand-right": 7,
}


def get_named_joint(joints: List, name: str, type="smpl"):
    """get SMPL joint by name

    Args:
        joints (List): list of SMPL joints
        name (str): joint to be extracted

    Returns:
        Tuple[float, float, float]: Coordinates of the selected joint
    """
    if type == "smpl":
        mapping = get_mapping_arr()
        index = joint_names_body_25[name]
        return joints[np.where(mapping == index)]
    if type == "body_25":
        return joints[joint_names_body_25[name]]


def get_indices_by_name(names: List[str], type="smpl"):
    return [get_index_by_name(name, type=type) for name in names]


def get_index_by_name(name: str, type="smpl"):
    if type == "smpl":
        mapping = get_mapping_arr()
        index = joint_names_body_25[name]
        return np.where(mapping == index)
    if type == "body_25":
        return joint_names_body_25[name]


def get_named_joints(joints: List, names: List[str], type="smpl"):
    return np.array([get_named_joint(joints, name, type=type) for name in names]).squeeze()


def apply_mapping(
        input_data: List,
        mapping: list):
    return [input_data[i] if i != -1 else (0, 0, 0) for i in mapping]


def openpose_to_opengl_coords(
    input_data: List[Tuple[float, float]],
    real_width: int,
    real_height: int
) -> (List[Tuple[float, float, float]], List[float]):
    """converts a list of OpenPose 2d keypoints with confidence to a opengl coordinate system 3d point list and a confidence array

    Args:
        input_data (List[Tuple[float, float]]): [description]
        real_width (int): OpenPose input image/data width
        real_height (int): OpenPose input image/data height


    Returns:
        [type]: [description]
    """

    points = np.array([
        [
            x / real_width * 2 - 1,
            -y / real_height * 2 + 1,
            0
        ] for (x, y, z) in input_data])

    conf = np.array([
        z for (_, _, z) in input_data
    ])

    return (points, conf)


def smpl_to_openpose(print_mapping: True):
    """Utility for remapping smpl mapping indices to openpose mapping indices. 

    Returns:
        [type]: [description]
    """
    mapping = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
               8, 1, 4, 7, 56, 57, 58, 59]

    arr = np.ones(127) * -1

    if print_mapping:
        for i, v in enumerate(mapping):
            arr[v] = i
            print(v, i)

        print('[')
        for i, v in enumerate(arr):
            print(int(v), ",")
        print(']')

    return arr
