from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import trimesh
import pyrender

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


def render_points(scene, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0], name=None):
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = colors
    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:, :3, 3] = points
    pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # return the render scene node
    return scene.add(pcl, name=name)
