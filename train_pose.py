from modules.distance_loss import WeightedMSELoss
from modules.utils import get_loss_layers
from camera_estimation import TorchCameraEstimate
import smplx
import torch
from tqdm import tqdm
import torchgeometry as tgm

# internal imports
from modules.pose import BodyPose
from modules.filter import JointFilter
from modules.camera import SimpleCamera
from renderer import Renderer


def train_pose(
    model: smplx.SMPL,
    # current datapoints
    keypoints,
    keypoint_conf,
    # 3D to 2D camera layer
    camera: SimpleCamera,

    # model type
    model_type="smplx",

    # pytorch config
    device=torch.device('cuda'),
    dtype=torch.float32,

    # optimizer settings
    optimizer=None,
    optimizer_type="LBFGS",
    learning_rate=1e-3,
    iterations=60,
    patience=10,

    # renderer options
    renderer: Renderer = None,
    render_steps=True,

    extra_loss_layers=[],

    use_progress_bar=True,
    use_openpose_conf_loss=True,
    loss_analysis=True
):
    if use_progress_bar:
        print("[pose] starting training")
        print("[pose] dtype=", dtype, device)

    offscreen_step_output = []

    # is enabled will use openpose keypoint confidence
    # as weights on the loss components
    if use_openpose_conf_loss:
        loss_layer = WeightedMSELoss(
            weights=keypoint_conf,
            device=device,
            dtype=dtype
        )
    else:
        loss_layer = torch.nn.MSELoss(reduction="sum").to(
            device=device,
            dtype=dtype
        )

    # make sure camera module is on the correct device
    camera = camera.to(device=device, dtype=dtype)

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)

    keypoint_filter = JointFilter(
        model_type=model_type, filter_dims=3).to(device=device, dtype=dtype)

    # filter keypoints
    keypoints = keypoint_filter(keypoints)

    # create filter layer to ignore unused joints, keypoints during optimization
    filter_layer = JointFilter(
        model_type=model_type, filter_dims=3).to(device=device, dtype=dtype)

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device,
                          useBodyMeanAngles=False).to(device=device, dtype=dtype)

    parameters = [model.global_orient, pose_layer.body_pose]

    # setup all loss layers
    for l in extra_loss_layers:
        # make sure layer is running on the correct device
        l.to(device=device, dtype=dtype)

        # register parameters if present
        if l.has_parameters:
            parameters = parameters + list(l.parameters())

    if optimizer is None:
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam

    optimizer = optimizer(parameters, learning_rate)

    if use_progress_bar:
        pbar = tqdm(total=iterations)

    # store results for optional plotting
    cur_patience = patience
    best_loss = None
    best_output = None

    # setup loss history data gathergin
    loss_history = []
    if loss_analysis:
        loss_components = {"points": []}
        for l in extra_loss_layers:
            loss_components[l.__class__.__name__] = []

    # prediction and loss computation closere
    def predict():
        # return joints based on current model state
        body_joints, cur_pose = pose_layer()

        # compute homogeneous coordinates and project them to 2D space
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()
        points = filter_layer(points)

        # compute loss between 2D joint projection and OpenPose keypoints
        loss = loss_layer(points, keypoints)
        if loss_analysis:
            loss_components['points'].append(loss.item())
        # apply extra losses
        for l in extra_loss_layers:
            cur_loss = l(cur_pose, body_joints, points,
                         keypoints, pose_layer.cur_out)
            if loss_analysis:
                loss_components[l.__class__.__name__].append(cur_loss.item())
            loss = loss + cur_loss
        return loss

    # main optimizer closure
    def optim_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = predict()

        if loss.requires_grad:
            loss.backward()
        return loss

    # main optimization loop
    for t in range(iterations):
        loss = optimizer.step(optim_closure)

        # compute loss
        cur_loss = loss.item()

        loss_history.append(loss)

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
            R = camera.trans.detach().cpu().numpy().squeeze()
            renderer.render_model_with_tfs(
                model, pose_layer.cur_out, keep_pose=True, transforms=R)

            if renderer.use_offscreen:
                offscreen_step_output.append(renderer.get_snapshot())
            # renderer.set_group_pose("body", R)

    if use_progress_bar:
        pbar.close()
        print("Final result:", loss.item())
    return best_output, loss_history, offscreen_step_output, loss_components


def train_pose_with_conf(
    config,
    camera: TorchCameraEstimate,
    model: smplx.SMPL,
    keypoints,
    keypoint_conf,
    device=torch.device('cpu'),
    dtype=torch.float32,
    renderer: Renderer = None,
    render_steps=True,
    use_progress_bar=True,
    print_loss_layers=False
):

    # configure PyTorch device and format
    # dtype = torch.float64
    if 'device' in config['pose'] is not None:
        device = torch.device(config['pose']['device'])
    else:
        device = torch.device('cpu')

    # create camera module
    pose_camera, cam_trans, cam_int, cam_params = SimpleCamera.from_estimation_cam(
        cam=camera,
        use_intrinsics=config['pose']['useCameraIntrinsics'],
        dtype=dtype,
        device=device,
    )

    # pose_camera, cam_trans = SimpleCamera.dummy_camera(
    #     device=device, dtype=dtype)

    # apply transform to scene
    if renderer is not None:
        renderer.set_group_pose("body", cam_trans.cpu().numpy())

    loss_layers = get_loss_layers(config, model, device, dtype)

    if print_loss_layers:
        print(loss_layers)

    best_output, loss_history, offscreen_step_output, loss_components = train_pose(
        model=model.to(dtype=dtype),
        keypoints=keypoints,
        keypoint_conf=keypoint_conf,
        camera=pose_camera,
        device=device,
        dtype=dtype,
        renderer=renderer,
        optimizer_type=config['pose']['optimizer'],
        iterations=config['pose']['iterations'],
        learning_rate=config['pose']['lr'],
        render_steps=render_steps,
        use_openpose_conf_loss=config['pose']['useOpenPoseConf'],
        use_progress_bar=use_progress_bar,
        extra_loss_layers=loss_layers
    )

    return best_output, cam_trans, loss_history, offscreen_step_output, loss_components
