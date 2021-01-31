from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import trimesh
import pyrender


def render_model(
    scene,
    model,
    model_out,
    color=[0.3, 0.3, 0.3, 0.8],
    name=None,
    replace=False,
    pose=None
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

    return scene.add(mesh, name=name, pose=pose)


def render_points(scene, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0], name=None):
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:, :3, 3] = points
    pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # return the render scene node
    return scene.add(pcl, name=name)


def render_camera(scene, radius=0.5, height=0.5, color=[0.0, 0.0, 1.0, 1.0], name=None):
    sm = trimesh.creation.cone(radius, height, sections=None, transform=None)
    sm.visual.vertex_colors = color
    tfs = np.eye(4)
    pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # return the render scene node
    return scene.add(pcl, name=name)


def render_image_plane(scene, image, scale, name=None):
    height, width, _ = image.shape
    mat = trimesh.visual.texture.TextureVisuals(
        image=image, uv=[[0, 0], [0, 1], [1, 0], [1, 1]])
    tm = trimesh.load('plane.obj', visual=mat)
    tm.visual = mat
    tfs = np.eye(4)
    tfs[0, 0] = width / height * scale
    tfs[1 ,1] *= scale
    tfs[2 ,2] *= scale
    tfs[0, 3] = (width / height - 1)* scale
    material2 = pyrender.Material(name=name, emissiveTexture=image)
    m = pyrender.Mesh.from_trimesh(tm, poses=tfs)
    return scene.add(m, name=name)
