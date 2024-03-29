import matplotlib.pyplot as plt
import torch
import numpy as np
import smplx
import os
from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser
from human_body_prior.tools.model_loader import load_vposer


def get_model_path(type, gender, dir):
    return os.path.join(
        dir,
        type,
        type.upper() + "_" +
        gender.upper() + ".npz")


def get_model_path_from_conf(config):
    return get_model_path(
        config['smpl']['type'],
        config['smpl']['gender'],
        config['smpl']['modelRootDir']
    )


class VPoserModel():
    global_vposer = None

    def __init__(
        self,
        model_type='smpl',
        body_model_path="./models/smplx/SMPLX_MALE.npz",
        vposer_model_path="./vposer_v1_0",
        ext='npz',
        gender='neutral',
        create_body_pose=True,
        plot_joints=True,
        num_betas=10,
        sample_shape=False,
        sample_expression=False,
        num_expression_coeffs=10,
        use_vposer=True
    ):
        self.body_model_path = body_model_path
        self.vposer_model_path = vposer_model_path
        self.model_type = model_type
        self.ext = ext
        self.gender = gender
        self.plot_joints = plot_joints
        self.num_betas = num_betas
        self.sample_shape = sample_shape
        self.sample_expression = sample_expression
        self.num_expression_coeffs = num_expression_coeffs
        self.create_body_pose = create_body_pose
        self.use_vposer = use_vposer

        self.create_model()

    def create_model(self):

        self.model = BodyModelWithPoser(
            bm_path=self.body_model_path,
            batch_size=1,
            poser_type="vposer",
            smpl_exp_dir=self.vposer_model_path
        )
        return self.model

    def get_vposer_latent(self):
        return self.model.poZ_body

    def get_pose(self):
        return self.model.pose_body

    def from_conf(config, use_global=True):
        model_path = get_model_path_from_conf(config)

        if VPoserModel.global_vposer is None:
            VPoserModel.global_vposer = VPoserModel(
                model_type=config['smpl']['type'],
                gender=config['smpl']['gender'],
                vposer_model_path=config['pose']['vposerPath'],
                body_model_path=model_path)

        return VPoserModel.global_vposer


class SMPLyModel():
    def __init__(
        self,
        model_folder,
        model_type='smplx',
        ext='npz',
        gender='male',
        create_body_pose=True,
        plot_joints=True,
        num_betas=10,
        sample_shape=False,
        sample_expression=True,
        num_expression_coeffs=10,
        use_face_contour=False,
        use_vposer_init=False,
        device=torch.device('cpu')
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
        self.use_face_contour = use_face_contour
        self.create_body_pose = create_body_pose
        self.use_vposer_init = use_vposer_init
        self.device = device

    def create_model(self, initial_pose=None):
        if self.use_vposer_init:
            # sample a valid human shape via vposer
            vp, ps = load_vposer("./vposer_v1_0")
            vp = vp.to(device=self.device)
            self.vp = vp
            self.vposer_sample = torch.from_numpy(
                np.random.randn(1, 32).astype(np.float32)).to(device=self.device)

            # sample SMPL body pose from vposer
            initial_pose = self.vp.decode(
                self.vposer_sample, output_type='aa').view(-1, 63)

        self.model = smplx.create(
            self.model_folder,
            model_type=self.model_type,
            gender=self.gender,
            body_pose=initial_pose,
            use_face_contour=self.use_face_contour,
            create_body_pose=self.create_body_pose,
            num_betas=self.num_betas,
            return_verts=True,
            num_expression_coeffs=self.num_expression_coeffs,
            ext=self.ext)
        return self.model

    def from_config(config, device=None, dtype=None):
        return SMPLyModel(
            model_folder=config['smpl']['modelRootDir'],
            gender=config['smpl']['gender'],
            model_type=config['smpl']['type'],
            ext=config['smpl']['ext'],
            use_vposer_init=config['smpl']['useVposerInit'],
            device=device
        )

    def model_from_conf(config, device=None, dtype=None, initial_pose=None):
        return SMPLyModel.from_config(config, device=device, dtype=dtype).create_model(initial_pose=initial_pose)
