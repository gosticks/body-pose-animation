from typing import List, Set, Dict, Tuple, Optional
from utils.mapping import get_named_joints
from renderer import DefaultRenderer, Renderer
import numpy as np
import cv2
import yaml
import os.path
import glob
from datetime import datetime


def getfilename_from_conf(config, index=None, include_date=True):
    """create a filename containing most training props

    Args:
        config ([type]): config object
        index ([type]): sample index
    """
    name = ""

    if include_date:
        now = datetime.now()
        name += now.strftime("%d-%m-%H-%M")

    if index is not None:
        name = str(index).zfill(3) + "-"
    name = name + config['pose']['optimizer']
    name = name + "-lr[" + str(config['pose']['lr']) + "]"
    name = name + "-it[" + str(config['pose']['iterations']) + "]"
    if config['pose']['anglePrior']['enabled']:
        name = name + \
            "-ap[" + str(config['pose']['anglePrior']['weight']) + "]"
    if config['pose']['bodyPrior']['enabled']:
        name = name + "-bp[" + str(config['pose']['bodyPrior']['weight']) + "]"
    if config['pose']['angleSumLoss']['enabled']:
        name = name + \
            "-as[" + str(config['pose']['angleSumLoss']['weight']) + "]"

    if config['pose']['temporal']['enabled']:
        name = name + "-temporal"

    return name


def get_output_path_from_conf(config, index=None):
    name = getfilename_from_conf(config, index=index)

    return os.path.join(config['output']['rootDir'], name)


def load_config(name=None):
    if name is None:
        config_name_env = os.getenv('CONFIG_PATH')
        if config_name_env is not None:
            name = config_name_env
        else:
            name = './config.yaml'

    with open(name) as file:
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


# Rename files, later save output of openpose to specific format so that this is not necessary
def rename_files(dir):
    idx = 0
    counter = 0
    for x in os.listdir(dir):
        _, ext = x.split('.')
        try:
            os.rename(dir + x, dir + str(counter) + "." + ext)
            idx += 1
            if idx == 2:
                counter += 1
                idx = 0
        except FileExistsError:
            print("Files already renamed")
            break

# Create a new filename by incrementing counter


def get_new_filename():
    conf = load_config()
    results_dir = conf['output']['rootDir']
    result_prefix = conf['output']['prefix']

    results = glob.glob(results_dir + "*.pkl")
    if len(results) == 0:
        return result_prefix + "0.pkl"
    else:
        latest_file = max(results, key=os.path.getctime)
        num = int(latest_file.split("-")[1].split(".")[0])
        return result_prefix + str(num + 1) + ".pkl"


def setup_training(model, dataset, sample_index, renderer=True, offscreen=False):
    keypoints, conf = dataset[sample_index]
    img_path = dataset.get_image_path(sample_index)

    # ---------------------------------
    # Generate model and get joints
    # ---------------------------------
    model_out = model()
    joints = model_out.joints.detach().cpu().numpy().squeeze()

    # ---------------------------------
    # Draw in the joints of interest
    # ---------------------------------
    est_scale = estimate_scale(joints, keypoints)

    # apply scaling to keypoints
    keypoints = keypoints * est_scale

    # integrating Camera Estimation

    init_joints = get_torso(joints)
    init_keypoints = get_torso(keypoints)

    r = None

    if renderer:
        # setup renderer
        r = DefaultRenderer(offscreen=offscreen)
        r.setup(
            model=model,
            model_out=model_out,
            keypoints=keypoints,
            init_keypoints=init_keypoints,
            init_joints=init_joints,
            img_path=img_path,
            img_scale=est_scale
        )
    return init_keypoints, init_joints, keypoints, conf, est_scale, r, img_path
