import matplotlib.pyplot as plt
import numpy as np
import smplx


class SMPLyModel():
    def __init__(
        self,
        model_folder,
        model_type='smpl',
        ext='npz',
        gender='neutral',
        create_body_pose=True,
        plot_joints=True,
        num_betas=10,
        sample_shape=False,
        sample_expression=True,
        num_expression_coeffs=10,
        plotting_module='pyrender',
        use_face_contour=False
    ):
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
        self.create_body_pose = create_body_pose

    def create_model(self):
        self.model = smplx.create(
            self.model_folder,
            model_type=self.model_type,
            gender=self.gender,
            use_face_contour=self.use_face_contour,
            create_body_pose=self.create_body_pose,
            num_betas=self.num_betas,
            return_verts=True,
            num_expression_coeffs=self.num_expression_coeffs,
            ext=self.ext)
        return self.model
