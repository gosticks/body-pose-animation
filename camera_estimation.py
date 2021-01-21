
# Initial camera estimation based on the torso keypoints obtained from OpenPose.

import yaml
from dataset import SMPLyDataset
from model import *
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time

dtype = torch.float64

def load_config():
    with open('./config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

# TODO: use already created methods
def create_visualization_points(points, color):
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (len(points_3d), 1, 1))
    tfs[:, :3, 3] = points
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    return joints_pcl


class CameraEstimate:
    def __init__(self, model: smplx.SMPLX, dataset):
        self.model = model
        self.dataset = dataset
        self.output_model = model(return_verts=True)

    def get_torso_keypoints(self):
        # TODO: Later use separate functions for normalizing and loading the keypoints
        keypoints = self.dataset[0]
        keypoints = np.reshape(keypoints, (25, 3))

        # TODO: use data loader methods
        torso_joints_idxs = [1, 2, 16, 17]  # hip left, hip right, left shoulder, right shoulder
        torso_keypoints_2d = np.array([keypoints[x] for x in torso_joints_idxs])
        torso_keypoints_2d[:, 0] = torso_keypoints_2d[:, 0] / 1920 * 2 - 1
        torso_keypoints_2d[:, 1] = torso_keypoints_2d[:, 1] / 1080 * 2 - 1
        torso_keypoints_2d[:, 2] = 0

        smpl_keypoints = self.output_model.joints.detach().cpu().numpy()
        torso_keypoints_3d = np.array([smpl_keypoints[0][x] for x in torso_joints_idxs])

        return torso_keypoints_2d, torso_keypoints_3d


    def visualize_mesh(self, points_2d, points_3d, pose):

        # hardcoded scaling factor
        points_3d /= 2.6

        self.scene = pyrender.Scene()

        vertices = self.output_model.vertices.detach().cpu().numpy().squeeze() / 2.6
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        tri_mesh = trimesh.Trimesh(vertices, self.model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.verts = self.scene.add(mesh)

        if pose is not None:
            self.scene.set_pose(self.verts, pose)

        color = [0.1, 0.9, 0.1, 1.0]
        self.scene.add(create_visualization_points(points_3d, color))

        color = [0.1, 0.1, 0.9, 1.0]
        self.transformed_points = self.scene.add(create_visualization_points(points_3d, color))

        if pose is not None:
            self.scene.set_pose(self.transformed_points, pose)

        color = [0.9, 0.1, 0.1, 1.0]
        self.scene.add(create_visualization_points(points_2d, color))
        pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True, viewport_size=(1280, 720))

    def loss_model(self, params, X):
        translation = params[:3]
        rotation = R.from_euler('xyz', [params[3], params[4], params[5]], degrees=False)
        y_pred = X @ rotation.as_matrix() + translation
        return y_pred

    def sum_of_squares(self, params, X, Y):
        y_pred = self.loss_model(params, X)
        loss = np.sum((y_pred - Y) ** 2)
        return loss

    def callback(self, params):
        time.sleep(0.3)
        #input("Press a key for next iteration...")
        current_pose = self.params_to_pose(params)
        self.scene.set_pose(self.transformed_points, current_pose)
        self.scene.set_pose(self.verts, current_pose)

    def params_to_pose(self, params):
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler('xyz', [params[3], params[4], params[5]], degrees=False).as_matrix()
        pose[:3, 3] = params[:3]
        return pose

    def estimate_camera_pos(self):

        translation = np.zeros(3)
        rotation = np.random.rand(3) * 2 * np.pi
        params = np.concatenate((translation, rotation))

        points_2d, points_3d = self.get_torso_keypoints()

        self.visualize_mesh(points_2d, points_3d, None)

        res = minimize(self.sum_of_squares, x0=params, args=(points_3d, points_2d), callback=self.callback, tol=1e-4, method="BFGS")
        print(res)

        pose = self.params_to_pose(res.x)
        print(pose)

        return pose

conf = load_config()
dataset = SMPLyDataset()

# TODO: use data loader
model = smplx.create("models/smpl/SMPL_FEMALE.pkl", model_type='smpl')
output_model = model(return_verts=True)

camera = CameraEstimate(model, dataset)
points_2d, points_3d = camera.get_torso_keypoints()
pose = camera.estimate_camera_pos()
