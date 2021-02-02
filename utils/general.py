from typing import List, Set, Dict, Tuple, Optional
from utils.mapping import get_named_joints
import numpy as np
import cv2
import yaml
import os.path


def load_config():
    with open('./config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def get_torso(joints):
    """get torso points from SMPL joint array

    Args:
        joints ([type]): SMPL joints array

    Returns:
        [type]: torso points array
    """
    cam_est_joints_names = ["hip-left", "hip-right",
                            "shoulder-left", "shoulder-right"]
    return get_named_joints(joints, cam_est_joints_names)


def estimate_scale(joints, keypoints, pairs=[
    ("shoulder-right", "hip-right"),
    ("shoulder-left", "hip-left")
], cam_fy=850):
    """estimate image depth based on the height changes due to perspective.
    This method only provides a rough estimate by computing shoulder to hip distances
    between SMPL joints and OpenPose keypoints.

    Args:
        joints ([type]): List of all SMPL joints
        keypoints ([type]): List of all OpenPose keypoints
        cam_fy (int, optional): Camera Y focal length. Defaults to 850.
    """

    # store distance vectors
    smpl_dists = []
    ops_dists = []

    for (j1, j2) in pairs:
        smpl_joints = get_named_joints(joints, [j1, j2])
        ops_keyp = get_named_joints(keypoints, [j1, j2])

        smpl_dists.append(smpl_joints[0] - smpl_joints[1])
        ops_dists.append(ops_keyp[0] - ops_keyp[1])

    smpl_height = np.linalg.norm(smpl_dists, axis=0).mean()
    ops_height = np.linalg.norm(ops_dists, axis=0).mean()

    return cam_fy / 1080 * smpl_height / ops_height


def estimate_focal_length(run_estimation: bool = False):
    """
    Estimate focal length by selecting a region of image whose real width is known.
    Executed once to compute a camera intrinsics matrix.
    For now, focal length = 850

    :return: focal_length
    """

    # TODO: adjust known distances with more precise values if this method works

    if run_estimation:
        image = cv2.imread(os.path.dirname(__file__) + '/../samples/003.png')
        cv2.imshow("image", image)
        marker = cv2.selectROI(
            "image", image, fromCenter=False, showCrosshair=True)

        # width of the selected region (object) in the image
        region_width = marker[2]

        # known real distance from the camera to the object
        known_distance = 200

        # known real width of the object
        known_width = 50

        focal_length = (region_width * known_distance) / known_width
        print("Focal length:", focal_length)
    else:
        focal_length = 850

    return focal_length
