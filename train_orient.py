from utils.mapping import get_indices_by_name
from modules.distance_loss import WeightedMSELoss
from modules.utils import get_loss_layers
from camera_estimation import TorchCameraEstimate
import smplx
import torch
import torch.nn as nn
from tqdm import tqdm
import torchgeometry as tgm

# internal imports
from modules.pose import BodyPose
from modules.filter import JointFilter
from modules.camera import SimpleCamera
from renderer import Renderer


def train_orient(
    model: smplx.SMPL,
    # current datapoints
    keypoints,
    # 3D to 2D camera layer
    camera: SimpleCamera,

    # pytorch config
    device=torch.device('cuda'),
    dtype=torch.float32,

    # optimizer settings
    optimizer=None,
    optimizer_type="Adam",
    learning_rate=1e-3,
    iterations=60,
    patience=10,

    joint_names=["hip-left", "hip-right",
                   "shoulder-left", "shoulder-right"],

    # renderer options
    renderer: Renderer = None,
    render_steps=True,

    use_progress_bar=True,
):
    if use_progress_bar:
        print("[pose] starting training")
        print("[pose] dtype=", dtype, device)

    loss_layer = torch.nn.MSELoss(reduction="sum").to(
        device=device,
        dtype=dtype
    )

    # make sure camera module is on the correct device
    camera = camera.to(device=device, dtype=dtype)

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)

    # torso indices
    torso_indices = get_indices_by_name(joint_names)

    torso_indices = torch.tensor(
        torso_indices, dtype=torch.int64, device=device).reshape(4)

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device,
                          useBodyMeanAngles=False).to(device=device, dtype=dtype)

    parameters = [model.global_orient]

    if use_progress_bar:
        pbar = tqdm(total=iterations)

    # store results for optional plotting
    cur_patience = patience
    best_loss = None
    best_output = None

    if optimizer is None:
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam

    optimizer = optimizer(parameters, learning_rate)

    # prediction and loss computation closere
    def predict():
        # return joints based on current model state
        body_joints, cur_pose = pose_layer()

        # compute homogeneous coordinates and project them to 2D space
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()

        # compute loss between 2D joint projection and OpenPose keypoints
        loss = loss_layer(points[torso_indices],
                          keypoints[torso_indices])

        return loss

    # main optimizer closure
    def optim_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = predict()

        if loss.requires_grad:
            loss.backward()
        return loss

    # camera translation
    R = camera.trans.detach().cpu().numpy().squeeze()

    # main optimization loop
    for t in range(iterations):
        loss = optimizer.step(optim_closure)

        # compute loss
        cur_loss = loss.item()

        if best_loss is None:
            best_loss = cur_loss
        elif cur_loss < best_loss:
            best_loss = cur_loss
            best_output = pose_layer.cur_out
        else:
            cur_patience = cur_patience - 1

        if patience == 0:
            print("[train] aborted due to patience limit reached")

        if use_progress_bar:
            pbar.set_description("Error %f" % cur_loss)
            pbar.update(1)

        if renderer is not None and render_steps:
            renderer.render_model(
                model=model,
                model_out=pose_layer.cur_out,
                transform=R
            )

    if use_progress_bar:
        pbar.close()
        print("Final result:", loss.item())
    return best_output


def train_orient_with_conf(
    config,
    camera_layer: TorchCameraEstimate,
    model: smplx.SMPL,
    keypoints,
    device=torch.device('cpu'),
    dtype=torch.float32,
    renderer: Renderer = None,
    render_steps=True,
    use_progress_bar=True,
):

    best_output = train_orient(
        model=model.to(dtype=dtype),
        keypoints=keypoints,
        camera=camera_layer,
        device=device,
        dtype=dtype,
        renderer=renderer,
        joint_names=config['orientation']['joint_names'],
        optimizer_type=config['orientation']['optimizer'],
        iterations=config['orientation']['iterations'],
        learning_rate=config['orientation']['lr'],
        render_steps=render_steps,
        use_progress_bar=use_progress_bar,
    )

    return best_output.global_orient
