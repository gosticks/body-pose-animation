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
    keypoints,
    keypoint_conf,
    camera: SimpleCamera,
    model_type="smplx",
    learning_rate=1e-3,
    device=torch.device('cuda'),
    dtype=torch.float32,
    renderer: Renderer = None,
    optimizer=None,
    optimizer_type="LBFGS",
    iterations=60,
    useBodyPrior=True,
    useAnglePrior=True,
    useConfWeights=True,
    patience=10,
    body_prior_weight=2,
    angle_prior_weight=0.5
):

    loss_layer = torch.nn.MSELoss()

    # setup keypoint data
    keypoints = torch.tensor(keypoints).to(device=device, dtype=dtype)
    # get a list of openpose conf values
    keypoints_conf = torch.tensor(keypoint_conf).to(device)

    # create filter layer to ignore unused joints, keypoints during optimization
    filter_layer = JointFilter(model_type=model_type, filter_dims=3)

    # setup torch modules
    pose_layer = BodyPose(model, dtype=dtype, device=device,
                          useBodyMeanAngles=useBodyPrior).to(device)

    parameters = [pose_layer.body_pose]

    # loss layers
    if useBodyPrior:
        vposer = VPoserModel()
        vposer_layer = vposer.model
        vposer_params = vposer.get_vposer_latent()
        parameters.append(vposer_params)

    if useAnglePrior:
        angle_prior_layer = AnglePriorsLoss(dtype=dtype, device=device)

    if optimizer is None:
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam

    optimizer = optimizer(parameters, learning_rate)

    pbar = tqdm(total=iterations)

    def predict():
        pose_extra = None

        if useBodyPrior:
            body = vposer_layer()
            poZ = body.poZ_body
            pose_extra = body.pose_body

        # return joints based on current model state
        body_joints = pose_layer(pose_extra)

        # compute homogeneous coordinates and project them to 2D space
        points = tgm.convert_points_to_homogeneous(body_joints)
        points = camera(points).squeeze()

        # filter out unused joints
        points = filter_layer(points)

        # compute loss between 2D joint projection and OpenPose keypoints

        if useConfWeights:
            distance = points - keypoints
            loss = distance * (keypoint_conf)
        else:
            loss = loss_layer(points, keypoints)

        if useBodyPrior:
            # apply pose prior loss.
            loss = loss + poZ.pow(2).sum() * body_prior_weight
        if useAnglePrior:
            loss = loss + \
                angle_prior_layer(pose_layer.body_pose) * angle_prior_weight

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

        # if t % 5 == 0:
        #     time.sleep(5)

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

        pbar.set_description("Error %f" % cur_loss)
        pbar.update(1)

        if renderer is not None:
            R = camera.trans.numpy().squeeze()
            renderer.render_model_with_tfs(
                model, pose_layer.cur_out, keep_pose=True, transforms=R)
            # renderer.set_group_pose("body", R)

    pbar.close()
    print("Final result:", loss.item())
    return pose_layer.cur_out


def train_pose_with_conf(
    config,
    model: smplx.SMPL,
    keypoints,
    keypoint_conf,
    camera: SimpleCamera,
    device=torch.device('cpu'),
    dtype=torch.float32,
    renderer: Renderer = None,
):
    return train_pose(
        model=model,
        keypoints=keypoints,
        keypoint_conf=keypoint_conf,
        camera=camera,
        device=device,
        dtype=dtype,
        renderer=renderer,
        useAnglePrior=config['pose']['anglePrior']['enabled'],
        useBodyPrior=config['pose']['bodyPrior']['enabled'],
        useConfWeights=config['pose']['confWeights']['enabled'],
        learning_rate=config['pose']['lr'],
        optimizer_type=config['pose']['optimizer'],
        iterations=config['pose']['iterations'],
        body_prior_weight=config['pose']['bodyPrior']['weight'],
        angle_prior_weight=config['pose']['anglePrior']['weight']
    )
