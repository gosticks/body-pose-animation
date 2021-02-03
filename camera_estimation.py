
# Initial camera estimation based on the torso keypoints obtained from OpenPose.

from utils.mapping import get_named_joints
from utils.general import get_torso, load_config
from numpy.core.fromnumeric import transpose
from torch.autograd import backward
from dataset import *
from model import *
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from renderer import *
import torchgeometry as tgm
from torchgeometry.core.conversions import rtvec_to_pose
import cv2
from tqdm import tqdm
import torch.nn.functional as F


class CameraEstimate:
    def __init__(
            self,
            model: smplx.SMPL,
            dataset,
            keypoints,
            renderer,
            image_path=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
            est_scale=1):
        self.model = model
        self.dataset = dataset
        self.output_model = model(return_verts=True)
        self.renderer = renderer
        self.dtype = dtype
        self.device = device
        self.image_path = image_path
        self.keypoints = keypoints
        self.scale = torch.tensor([est_scale, est_scale, est_scale],
                                  requires_grad=False, dtype=self.dtype, device=self.device)

    def get_torso_keypoints(self):
        smpl_keypoints = self.output_model.joints.detach().cpu().numpy().squeeze()

        torso_keypoints_3d = get_torso(smpl_keypoints).reshape(4, 3)
        torso_keypoints_2d = get_torso(self.keypoints).reshape(4, 3)
        return torso_keypoints_2d, torso_keypoints_3d

    def visualize_mesh(self, keypoints, smpl_points):

        # hardcoded scaling factor
        # scaling_factor = 1
        # smpl_points /= scaling_factor

        # color_3d = [0.1, 0.9, 0.1, 1.0]
        # self.transformed_points = self.renderer.render_points(
        #     smpl_points, color=color_3d, name="smpl_torso", group_name="body")

        # color_2d = [0.9, 0.1, 0.1, 1.0]
        # self.renderer.render_keypoints(keypoints, color=color_2d)

        # model_color = [0.3, 0.3, 0.3, 0.8]
        # self.renderer.render_model(
        #     self.model, self.output_model, model_color)

        camera_color = [0.0, 0.0, 0.1, 1.0]
        self.camera_renderer = self.renderer.render_camera(color=camera_color)

        # if self.image_path is not None:
        #     self.renderer.render_image_from_path(self.image_path)
        self.renderer.start()

    def setup_visualization(self, render_points, render_keypoints):
        self.transformed_points = render_points

    def sum_of_squares(self, params, X, Y):
        y_pred = self.loss_model(params, X)
        loss = np.sum((y_pred - Y) ** 2)
        return loss

    def iteration_callback(self, params):
        time.sleep(0.1)
        #input("Press a key for next iteration...")
        current_pose = self.params_to_pose(params)

        # TODO: use renderer.py methods
        self.renderer.scene.set_group_pose("body", current_pose)
        # self.renderer.scene.set_pose(self.verts, current_pose)

    def params_to_pose(self, params):
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler(
            'xyz', [params[3], params[4], params[5]], degrees=False).as_matrix()
        pose[:3, 3] = params[:3]
        return pose

    def estimate_camera_pos(self):
        translation = np.zeros(3)
        rotation = np.random.rand(3) * 2 * np.pi
        params = np.concatenate((translation, rotation))
        print(params)

        init_points_2d, init_points_3d = self.get_torso_keypoints()

        self.visualize_mesh(init_points_2d, init_points_3d)

        res = minimize(self.sum_of_squares, x0=params, args=(init_points_3d, init_points_2d),
                       callback=self.iteration_callback, tol=1e-4, method="BFGS")
        print(res)

        transform_matrix = self.params_to_pose(res.x)
        return transform_matrix


class TorchCameraEstimate(CameraEstimate):
    def estimate_camera_pos(self):
        self.memory = None
        translation = torch.zeros(
            1, 3, requires_grad=True, dtype=self.dtype, device=self.device)
        rotation = torch.rand(1, 3, requires_grad=True,
                              dtype=self.dtype, device=self.device)
        rotation.float()
        translation.float()

        init_points_2d, init_points_3d = self.get_torso_keypoints()

        #self.visualize_mesh(init_points_2d, init_points_3d)

        init_points_2d = torch.from_numpy(init_points_2d).to(
            device=self.device, dtype=self.dtype)
        init_points_3d = torch.from_numpy(init_points_3d).to(
            device=self.device, dtype=self.dtype)
        init_points_3d_prepared = torch.ones(4, 4, 1).to(
            device=self.device, dtype=self.dtype)
        init_points_3d_prepared[:, :3, :] = init_points_3d.unsqueeze(
            0).transpose(0, 1).transpose(1, 2)

        params = [translation, rotation]
        opt = torch.optim.Adam(params, lr=0.1)

        loss_layer = torch.nn.MSELoss()

        stop = True
        tol = 3e-4
        print("Estimating Initial transform...")
        pbar = tqdm(total=100)
        current = 0
        while stop:
            y_pred = self.C(params, init_points_3d_prepared)
            loss = loss_layer(init_points_2d, y_pred)

            with torch.no_grad():
                opt.zero_grad()
                loss.backward()
                opt.step()
                current_pose = self.torch_params_to_pose(params)

                current_pose = current_pose.detach().numpy()

                if self.renderer is not None:
                    self.renderer.set_group_pose("body", current_pose)
                per = int((tol/loss*100).item())
                if per > 100:
                    pbar.update(abs(100 - current))
                    current = 100
                else:
                    pbar.update(per - current)
                    current = per
                stop = loss > tol

                if stop == True:
                    stop = self.patience_module(loss, 5)
        pbar.update(abs(100 - current))
        pbar.close()
        self.memory = None
        transform_matrix = self.torch_params_to_pose(params)
        current_pose = transform_matrix.detach().numpy()

        camera_translation = torch.tensor(
            [[0.5, 0.5, 5.0]], requires_grad=True, dtype=self.dtype, device=self.device)
        # camera_translation[0,2] = 5 * torch.ones(1)

        camera_rotation = torch.tensor(
            [[1e-5, 1e-5, 1e-5]], requires_grad=True, dtype=self.dtype, device=self.device)
        camera_intrinsics = torch.zeros(
            4, 4, dtype=self.dtype, device=self.device)
        camera_intrinsics[0, 0] = 5
        camera_intrinsics[1, 1] = 5
        camera_intrinsics[2, 2] = 1
        camera_intrinsics[0, 2] = 0.5
        camera_intrinsics[1, 2] = 0.5
        camera_intrinsics[3, 3] = 1

        params = [camera_translation, camera_rotation, camera_intrinsics]

        camera_extrinsics = self.torch_params_to_pose(params)

        # camera = tgm.PinholeCamera(camera_intrinsics.unsqueeze(0), camera_extrinsics.unsqueeze(0), torch.ones(1), torch.ones(1))

        init_points_3d_prepared = transform_matrix @ init_points_3d_prepared

        # result = self.transform_3d_to_2d(params, transform_matrix @ init_points_3d_prepared)

        opt2 = torch.optim.Adam(params, lr=0.1)

        stop = True
        first = True
        cam_tol = 6e-3
        print("Estimating Camera transformations...")
        pbar = tqdm(total=100)
        current = 0

        while stop:
            y_pred = self.transform_3d_to_2d(
                params, init_points_3d_prepared)
            loss = torch.nn.SmoothL1Loss()(init_points_2d.float(), y_pred.float())
            loss.requres_grad = True
            opt2.zero_grad()

            if first:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            opt2.step()
            #self.renderer.scene.set_pose(
                #self.camera_renderer, self.torch_params_to_pose(params).detach().numpy())
            per = int((cam_tol/loss*100).item())

            if per > 100:
                pbar.update(100 - current)
            else:
                pbar.update(per - current)

            current = per
            stop = loss > cam_tol

            if stop == True:
                stop = self.patience_module(loss, 5)

        pbar.update(100 - current)
        pbar.close()
        camera_transform_matrix = self.torch_params_to_pose(
            params)
        return camera_intrinsics, transform_matrix, camera_transform_matrix

    def transform_3d_to_2d(self, params, X):
        camera_ext = rtvec_to_pose(
            torch.cat((params[1], params[0])).view(-1).unsqueeze(0)).squeeze(0)
        camera_int = params[2]
        result = camera_int[:3, :3] @ camera_ext[:3, :] @ X
        result = result.squeeze(2)
        denomiator = torch.zeros(4, 3)
        denomiator[0, :] = result[0, 2]
        denomiator[1, :] = result[1, 2]
        denomiator[2, :] = result[2, 2]
        denomiator[3, :] = result[3, 2]
        result = result/denomiator
        result[:, 2] = 0
        return result

    def torch_params_to_pose(self, params):
        transform = rtvec_to_pose(
            torch.cat((params[1], params[0])).view(-1).unsqueeze(0))
        for i in range(3):
            transform[0, i, i] *= self.scale[i]
        return transform[0, :, :]

    def C(self, params, X):
        Ext_mat = rtvec_to_pose(
            torch.cat((params[1], params[0])).view(-1).unsqueeze(0))
        for i in range(3):
            Ext_mat[0, i, i] *= self.scale[i]
        y_pred = Ext_mat @ X
        y_pred = y_pred.squeeze(2)
        y_pred = y_pred[:, :3]
        return y_pred

    def loss_model(self, params, points):
        translation = params[:3]
        rotation = R.from_euler(
            'xyz', [params[3], params[4], params[5]], degrees=False)
        y_pred = points @ rotation.as_matrix() + translation
        return y_pred

    def patience_module(self, variable, counter: int):
        if self.memory is None:
            self.memory = torch.clone(variable)
            self.patience_count = 0
            return True
        if self.patience_count >= counter:
            self.memory is None
            self.patience_count = 0
            return False
        else:
            if torch.isclose(variable, self.memory).item():
                self.patience_count += 1
                return True
            else:
                self.patience_count = 0
                self.memory = torch.clone(variable)
                return True

# sample_index = 0

# conf = load_config()
# dataset = SMPLyDataset()
# model = SMPLyModel(conf['modelPath']).create_model()
# keypoints, conf = dataset[sample_index]
# camera = TorchCameraEstimate(
#     model,
#     dataset=dataset,
#     keypoints=keypoints,
#     renderer=Renderer(),
#     device=torch.device('cpu'),
#     dtype=torch.float32,
#     image_path="./samples/" + str(sample_index + 1).zfill(3) + ".png"

# )
# pose, transform = camera.estimate_camera_pos()
# print("Pose matrix: \n", pose)

# print("Transform matrix: \n", transform)
