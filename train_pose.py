from camera_estimation import TorchCameraEstimate
from modules.angle_clip import AngleClipper
from modules.angle import AnglePriorsLoss
import smplx
import torch
from tqdm import tqdm
import torchgeometry as tgm

# internal imports
from modules.pose import BodyPose
from modules.filter import JointFilter
from model import VPoserModel
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
    # configure loss function
    useBodyPrior=False,
    body_prior_weight=2,

    useAnglePrior=False,
    angle_prior_weight=0.5,

    use_angle_sum_loss=False,
    angle_sum_weight=0.1,

    body_mean_loss=False,
    body_mean_weight=0.01,

    useConfWeights=False,

    # renderer options
    renderer: Renderer = None,
    render_steps=True,

    vposer=None,

    use_progress_bar=True
):
    if use_progress_bar:
        print("[pose] starting training")
        print("[pose] dtype=", dtype, device)

    offscreen_step_output = []

    loss_layer = torch.nn.MSELoss().to(device=device, dtype=dtype)  # MSELoss()

    clip_loss_layer = AngleClipper().to(device=device, dtype=dtype)

    # make sure camera module is on the correct device
    camera = camera.to(device=device, dtype=dtype)

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)
    # get a list of openpose conf values
    # keypoints_conf = torch.tensor(keypoint_conf).to(device=device, dtype=dtype)

    # create filter layer to ignore unused joints, keypoints during optimization
    filter_layer = JointFilter(
        model_type=model_type, filter_dims=3).to(device=device, dtype=dtype)

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device,
                          useBodyMeanAngles=False).to(device=device, dtype=dtype)

    parameters = [pose_layer.body_pose]

    # loss layers
    if useBodyPrior:
        # TODO: handle this in vposer model
        vposer.model.to(device=device, dtype=dtype)
        latent_body = vposer.get_pose()
        latent_pose = vposer.get_vposer_latent()

        parameters.append(latent_pose)

    if useAnglePrior:
        angle_prior_layer = AnglePriorsLoss(
            dtype=dtype, device=device)

    if optimizer is None:
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam

    optimizer = optimizer(parameters, learning_rate)

    if use_progress_bar:
        pbar = tqdm(total=iterations)

    def predict():
        # pose_extra = None

        # if useBodyPrior:
        # body = vposer_layer()
        # poZ = body.poZ_body
        # pose_extra = body.pose_body

        # return joints based on current model state
        body_joints, cur_pose = pose_layer()

        # compute homogeneous coordinates and project them to 2D space
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()
        points = filter_layer(points)

        # compute loss between 2D joint projection and OpenPose keypoints

        if useConfWeights:
            distance = points - keypoints
            loss = distance * (keypoint_conf)
        else:
            loss = loss_layer(points, keypoints)

        body_mean_loss = 0.0
        if body_mean_loss:
            body_mean_loss = (cur_pose -
                              body_mean_pose).pow(2).sum() * body_mean_weight

        body_prior_loss = 0.0
        if useBodyPrior:
            # apply pose prior loss.
            body_prior_loss = latent_pose.pow(
                2).sum() * body_prior_weight

        angle_prior_loss = 0.0
        if useAnglePrior:
            angle_prior_loss = torch.sum(
                angle_prior_layer(cur_pose)) * angle_prior_weight
            angle_prior_loss

        angle_sum_loss = 0.0
        if use_angle_sum_loss:
            angle_sum_loss = clip_loss_layer(cur_pose) * angle_sum_weight

        loss = loss + body_mean_loss + body_prior_loss + angle_prior_loss + angle_sum_loss

        return loss

    def optim_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = predict()

        if loss.requires_grad:
            loss.backward()
        return loss

    # store results for optional plotting
    cur_patience = patience
    best_loss = None
    best_pose = None

    loss_history = []

    for t in range(iterations):
        optimizer.step(optim_closure)

        # LBFGS does not return the result, therefore we should rerun the model to get it
        with torch.no_grad():
            pred = predict()
            loss = optim_closure()

        # compute loss
        cur_loss = loss.item()

        loss_history.append(loss)

        if best_loss is None:
            best_loss = cur_loss
        elif cur_loss < best_loss:
            best_loss = cur_loss
            best_pose = pose_layer.cur_out
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
    return best_pose, loss_history, offscreen_step_output


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
    use_progress_bar=True
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

    # apply transform to scene
    if renderer is not None:
        renderer.set_group_pose("body", cam_trans.cpu().numpy())

    vposer = VPoserModel.from_conf(config)

    best_pose, loss_history, offscreen_step_output = train_pose(
        model=model.to(dtype=dtype),
        keypoints=keypoints,
        keypoint_conf=keypoint_conf,
        camera=pose_camera,
        device=device,
        dtype=dtype,
        renderer=renderer,
        useAnglePrior=config['pose']['anglePrior']['enabled'],
        useBodyPrior=config['pose']['bodyPrior']['enabled'],
        useConfWeights=config['pose']['confWeights']['enabled'],
        learning_rate=config['pose']['lr'],
        optimizer_type=config['pose']['optimizer'],
        iterations=config['pose']['iterations'],
        vposer=vposer,
        body_prior_weight=config['pose']['bodyPrior']['weight'],
        angle_prior_weight=config['pose']['anglePrior']['weight'],
        body_mean_loss=config['pose']['bodyMeanLoss']['enabled'],
        body_mean_weight=config['pose']['bodyMeanLoss']['weight'],
        use_angle_sum_loss=config['pose']['angleSumLoss']['enabled'],
        angle_sum_weight=config['pose']['angleSumLoss']['weight'],
        render_steps=render_steps,
        use_progress_bar=use_progress_bar
    )

    return best_pose, cam_trans, loss_history, offscreen_step_output
