from typing import List, Set, Dict, Tuple, Optional
import numpy as np


def get_mapping_arr(
    input_format: str = "body_25",
    output_format: str = "smpl",
) -> list:
    # TODO: expand features as needed
    # based on mappings found here
    # https://github.com/ortegatron/playing_smplifyx/blob/master/smplifyx/utils.py
    return np.array([
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
    ])


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
            y / real_height * 2,
            0
        ] for (x, y, z) in input_data])

    conf = np.array([
        z for (_, _, z) in input_data
    ])

    return (points, conf)
