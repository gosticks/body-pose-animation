from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import trimesh
import pyrender
import cv2

openpose_to_smpl = np.array([
    8,  # hip - middle
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


def get_mapping_arr(
    input_format: str = "body_25",
    output_format: str = "smpl",
) -> list:
    # TODO: expand features as needed
    # based on mappings found here
    # https://github.com/ortegatron/playing_smplifyx/blob/master/smplifyx/utils.py
    return openpose_to_smpl


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


def get_named_joints(joints: List, names: List[str], type="smpl"):
    return [get_named_joint(joints, name, type=type) for name in names]


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


def render_model(
    scene,
    model,
    model_out,
    color=[0.3, 0.3, 0.3, 0.8],
    name=None,
    replace=False,
):
    print(color)
    vertices = model_out.vertices.detach().cpu().numpy().squeeze()

    # set vertex colors, maybe use this to highlight accuracies
    vertex_colors = np.ones([vertices.shape[0], 4]) * color

    # triangulate vertex mesh
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    if name is not None and replace:
        for node in scene.get_nodes(name=name):
            scene.remove_node(node)

    return scene.add(mesh, name=name)


def render_points(scene, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0], name=None):
    print(color)
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:, :3, 3] = points
    pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # return the render scsene node
    return scene.add(pcl, name=name)


def estimate_scale(joints, keypoints, pairs=[
    ("shoulder-right", "hip-right"),
    ("shoulder-left", "hip-left")
], cam_fy=1):
    """estimate image depth based on the height changes due to perspective.
    This method only provides a rough estimate by computing shoulder to hip distances
    between SMPL joints and OpenPose keypoints.

    Args:
        joints ([type]): List of all SMPL joints
        keypoints ([type]): List of all OpenPose keypoints
        cam_fy (int, optional): Camera Y focal length. Defaults to 1.
    """

    # store distance vectors
    smpl_dists = []
    ops_dists = []

    for (j1, j2) in pairs:
        smpl_joints = get_named_joints(joints, [j1, j2])
        ops_keyp = get_named_joints(keypoints, [j1, j2])

        smpl_dists.append(smpl_joints[0] - smpl_joints[1])
        ops_dists.append(ops_keyp[0] - ops_keyp[1])

    smpl_height = np.linalg.norm(smpl_dists, axis=1).mean()
    ops_height = np.linalg.norm(ops_dists, axis=1).mean()

    return cam_fy * smpl_height / ops_height

def estimate_focal_length(run_estimation: bool = False):
    """
    Estimate focal length by selecting a region of image whose real width is known.
    Executed once to compute a camera intrinsics matrix.
    For now, focal length = 1000

    :return: focal_length
    """

    # TODO: adjust known distances with more precise values if this method works

    if run_estimation:
        image = cv2.imread('samples/001.jpg')
        cv2.imshow("image", image)
        marker = cv2.selectROI("image", image, fromCenter=False, showCrosshair=True)

        # width of the selected region (object) in the image
        region_width = marker[2]

        # known real distance from the camera to the object
        known_distance = 200

        # known real width of the object
        known_width = 50

        focal_length = (region_width * known_distance) / known_width
        print("Focal length:", focal_length)
    else:
        focal_length = 1000

    return focal_length
