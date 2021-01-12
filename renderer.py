
import smplx
import torch
import numpy as np
import trimesh
import pyrender


class SMPLyRenderer():
    def __init__(
        self,
    ):
        print("TODO: do some preparation here")

    # TODO: use __call__ for this
    def render_model(
        self,
        model: smplx.SMPL,
        betas: torch.TensorType,
        body_pose: torch.TensorType,
    ):
        # compute model
        model_out = model(
            betas=betas,
            body_pose=body_pose,
            return_verts=True,
        )

        # TODO: check if this also works with CUDA
        vertices = model_out.vertices.detach().cpu().numpy().squeeze()
        joints = model_out.joints.detach().cpu().numpy().squeeze()

        # set vertex colors, maybe use this to highlight accuracies
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        # triangulate vertex mesh
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        return (tri_mesh, joints, vertices)

    def display_mesh(
        self,
        tri_mesh: trimesh.Trimesh,
        joints=None,
        keypoints=None,
        render_openpose_wireframe=True,
    ):

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if joints is not None:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        if keypoints is not None:
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
            tfs = np.tile(np.eye(4), (len(keypoints), 1, 1))
            tfs[:, :3, 3] = keypoints
            keypoints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(keypoints_pcl)

        pyrender.Viewer(scene,
                        use_raymond_lighting=True,
                        # show_world_axis=True
                        )

    def display_model(
        self,
        model: smplx.SMPL,
        betas: torch.TensorType,
        body_pose: torch.TensorType,
        keypoints=None,
    ):
        (tri_mesh, joints, vertices) = self.render_model(model, betas, body_pose)

        self.display_mesh(tri_mesh, joints, keypoints)
