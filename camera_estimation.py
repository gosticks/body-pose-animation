
# Initial camera estimation based on the torso keypoints obtained from OpenPose.

from numpy.core.fromnumeric import transpose
from torch.autograd import backward
import yaml
from dataset import *
from model import *
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from utils import *
from renderer import *
import torchgeometry as tgm
from torchgeometry.core.conversions import rtvec_to_pose

dtype = torch.float64

def load_config():
    with open('./config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

class CameraEstimate:
    def __init__(self, model: smplx.SMPLX, dataset, renderer):
        self.model = model
        self.dataset = dataset
        self.output_model = model(return_verts=True)
        self.renderer = renderer
        self.toggle = False

    def get_torso_keypoints(self):

        keypoints = self.dataset[0]
        cam_est_joints_names = ["hip-left", "hip-right",
                                "shoulder-left", "shoulder-right"]

        smpl_keypoints = self.output_model.joints.detach().cpu().numpy().squeeze()

        torso_keypoints_3d = np.array(get_named_joints(smpl_keypoints, cam_est_joints_names))
        torso_keypoints_2d = np.array(get_named_joints(keypoints[0], cam_est_joints_names))

        return np.reshape(torso_keypoints_2d, (4, 3)), np.reshape(torso_keypoints_3d, (4, 3))


    def visualize_mesh(self, keypoints, smpl_points):

        # hardcoded scaling factor
        scaling_factor = 1
        smpl_points /= scaling_factor

        color_3d = [0.1, 0.9, 0.1, 1.0]
        self.transformed_points = self.renderer.render_points(smpl_points, color=color_3d)

        color_2d = [0.9, 0.1, 0.1, 1.0]
        self.renderer.render_keypoints(keypoints, color=color_2d)

        model_color = [0.3, 0.3, 0.3, 0.8]
        self.verts = self.renderer.render_model(self.model, self.output_model, model_color)

        camera_color = [0.0, 0.0, 0.1, 1.0]
        self.camera_renderer = self.renderer.render_camera( color=camera_color)

        img = cv2.imread("samples/001.jpg")

        self.renderer.render_image(img)
        self.renderer.start()

    def loss_model(self, params, points):
        translation = params[:3]
        rotation = R.from_euler('xyz', [params[3], params[4], params[5]], degrees=False)
        y_pred = points @ rotation.as_matrix() + translation
        return y_pred

    def sum_of_squares(self, params, X, Y):
        y_pred = self.loss_model(params, X)
        loss = np.sum((y_pred - Y) ** 2)
        return loss

    def iteration_callback(self, params):
        time.sleep(0.1)
        #input("Press a key for next iteration...")
        current_pose = self.params_to_pose(params)

        # TODO: use renderer.py methods
        self.renderer.scene.set_pose(self.transformed_points, current_pose)
        self.renderer.scene.set_pose(self.verts, current_pose)

    def params_to_pose(self, params):
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler('xyz', [params[3], params[4], params[5]], degrees=False).as_matrix()
        pose[:3, 3] = params[:3]
        return pose

    def torch_params_to_pose(self, params):
        transform = rtvec_to_pose(torch.cat((params[1],params[0])).view(-1).unsqueeze(0))
        return transform[0,:,:]

    def C(self, params, X):
        Ext_mat = rtvec_to_pose(torch.cat((params[1],params[0])).view(-1).unsqueeze(0))
        y_pred = Ext_mat @ X
        y_pred  = y_pred.squeeze(2)
        y_pred = y_pred[:,:3]
        return y_pred

    def transform_3d_to_2d(self, params, X):
        camera_ext = rtvec_to_pose(torch.cat((params[1],params[0])).view(-1).unsqueeze(0)).squeeze(0)
        camera_int = params[2]
        result = camera_int[:3,:3] @ camera_ext[:3,:] @ X
        result = result.squeeze(2)
        denomiator = torch.zeros(4,3)
        denomiator[0,:] = result[0,2]
        denomiator[1,:] = result[1,2]
        denomiator[2,:] = result[2,2]
        denomiator[3,:] = result[3,2]
        result = result/denomiator
        result[:,2] = 0
        return result

    def estimate_camera_pos(self):
        if self.toggle:
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
        else:
            translation = torch.zeros(1, 3, requires_grad = True)
            rotation = torch.rand(1, 3, requires_grad = True)
            rotation.float()
            translation.float()

            init_points_2d, init_points_3d = self.get_torso_keypoints()

            self.visualize_mesh(init_points_2d, init_points_3d)

            init_points_2d = torch.from_numpy(init_points_2d)
            init_points_3d = torch.from_numpy(init_points_3d)
            init_points_3d_prepared = torch.ones(4,4,1)
            init_points_3d_prepared[ : , :3 ,:] = init_points_3d.unsqueeze(0).transpose(0,1).transpose(1,2)

            params = [translation, rotation]
            opt = torch.optim.Adam(params, lr=0.1)

            stop = True
            while stop:
                y_pred = self.C(params, init_points_3d_prepared)
                loss = torch.nn.MSELoss()(init_points_2d.float(), y_pred.float() )
                loss.requres_grad = True
                opt.zero_grad()
                loss.float()
                loss.backward()
                opt.step()
                stop = loss > 3e-4
                current_pose = self.torch_params_to_pose(params)
                current_pose = current_pose.detach().numpy()
                self.renderer.scene.set_pose(self.transformed_points, current_pose)
                self.renderer.scene.set_pose(self.verts, current_pose)
            transform_matrix = self.torch_params_to_pose(params)
            current_pose = transform_matrix.detach().numpy()

            camera_translation = torch.tensor([[0.5,0.5,5.0]], requires_grad=True)
            # camera_translation[0,2] = 5 * torch.ones(1)
            
            camera_rotation = torch.tensor([[1e-5,1e-5,1e-5]], requires_grad=True)
            camera_intrinsics = torch.zeros(4,4)
            camera_intrinsics[0,0] = 5
            camera_intrinsics[1,1] = 5
            camera_intrinsics[2,2] = 1
            camera_intrinsics[0,2] = 0.5
            camera_intrinsics[1,2] = 0.5

            params = [camera_translation, camera_rotation, camera_intrinsics]
            
            camera_extrinsics = self.torch_params_to_pose(params)

            # camera = tgm.PinholeCamera(camera_intrinsics.unsqueeze(0), camera_extrinsics.unsqueeze(0), torch.ones(1), torch.ones(1))

            init_points_3d_prepared = transform_matrix @ init_points_3d_prepared

            # result = self.transform_3d_to_2d(params, transform_matrix @ init_points_3d_prepared)

            opt2 = torch.optim.Adam(params, lr=0.1)

            stop = True
            first = True
            while stop:
                y_pred = self.transform_3d_to_2d(params, init_points_3d_prepared) 
                loss = torch.nn.SmoothL1Loss()(init_points_2d.float(), y_pred.float())
                loss.requres_grad = True
                opt2.zero_grad()
                if first:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                opt2.step()
                stop = loss > 6e-5
                self.renderer.scene.set_pose(self.camera_renderer, self.torch_params_to_pose(params).detach().numpy())
                print(camera_translation, camera_rotation, loss)

            return transform_matrix


    
            

conf = load_config()
dataset = SMPLyDataset()
model = SMPLyModel(conf['modelPath']).create_model()

camera = CameraEstimate(model, dataset, Renderer())
pose = camera.estimate_camera_pos()
print("Pose matrix: \n", pose)
