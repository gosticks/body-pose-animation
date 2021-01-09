import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import os
import json
import numpy as np
import smplx
import pyrender
import trimesh
import torch

# TODO: clean this up after prototyping phase, currently this file will just house everything
class DataLoader():
    def __init__(
        self, 
        data_path,
        model_folder,
        model_type='smplx',
        ext='npz',
        gender='neutral',
        plot_joints=False,
        num_betas=10,
        sample_shape=True,
        sample_expression=True,
        num_expression_coeffs=10,
        plotting_module='pyrender',
        use_face_contour=False
        ):
        print("DataLoader: input location" + data_path)
        self.data_path = data_path
        self.model_folder = model_folder
        self.model_type = model_type
        self.ext = ext
        self.gender = gender
        self.plot_joints = plot_joints
        self.num_betas = num_betas
        self.sample_shape = sample_shape
        self.sample_expression = sample_expression
        self.num_expression_coeffs = num_expression_coeffs
        self.plotting_module = plotting_module
        self.use_face_contour = use_face_contour

    # display current sample that is loaded
    # only used for debug while developing
    def show_cur_item(self):
        img = mpimg.imread(os.path.join(self.data_path, 'frame-070.jpg'))
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img)
        self.draw_keypoints(ax)
        plt.show()

    # load people data from a openpose json dump
    def load_people(self):
        path = os.path.join(self.data_path, 'input_000000000068_keypoints.json')
        with open(path) as file:
            json_data = json.load(file)
            people = json_data['people']
        return people

    def create_model(self):
        model = smplx.create(
            self.model_folder,
            model_type=self.model_type,
            gender=self.gender,
            use_face_contour=self.use_face_contour,
            num_betas=self.num_betas,
            num_expression_coeffs=self.num_expression_coeffs,
            ext=self.ext)
        print(model)

        betas, expression = None, None
        if self.sample_shape:
            betas = torch.randn([1, model.num_betas], dtype=torch.float32)
        if self.sample_expression:
            expression = torch.randn(
                [1, model.num_expression_coeffs], dtype=torch.float32)

        output = model(betas=betas, expression=expression,
                    return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        print('Vertices shape =', vertices.shape)
        print('Joints shape =', joints.shape)


        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if self.plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


    # render keypoints on top of current image
    def draw_keypoints(self, plot):
        for p in self.load_people():
            keypoints = np.array(p['pose_keypoints_2d']).reshape(-1, 3)

            for point in keypoints:
                # draw points in image
                marker = Circle((point[0], point[1]), 5 * point[2])
                plot.add_patch(marker)
        print("drawing keypoints")
