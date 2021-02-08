# library imports
import pickle
import torch
from utils.video import make_video
from tqdm.auto import trange

# local imports
from train_pose import train_pose_with_conf
from model import SMPLyModel
from utils.general import get_new_filename, setup_training
from camera_estimation import TorchCameraEstimate


def optimize_sample(sample_index, dataset, config, device=torch.device('cpu'), dtype=torch.float32, interactive=True, offscreen=False, verbose=True):
    # prepare data and SMPL model
    model = SMPLyModel.model_from_conf(config)
    init_keypoints, init_joints, keypoints, conf, est_scale, r, img_path = setup_training(
        model=model,
        renderer=interactive,
        dataset=dataset,
        sample_index=sample_index,
        offscreen=offscreen
    )

    camera = TorchCameraEstimate(
        model,
        keypoints=keypoints,
        renderer=r,
        device=device,
        dtype=dtype,
        image_path=img_path,
        est_scale=est_scale,
        verbose=verbose,
        use_progress_bar=verbose
    )

    camera_transformation, camera_int, camera_params = camera.get_results(
        visualize=False)

    if not offscreen and interactive:
        # render camera to the scene
        camera.setup_visualization(r.init_keypoints, r.keypoints)

    # train for pose
    pose, loss_history, step_imgs = train_pose_with_conf(
        config=config,
        model=model,
        keypoints=keypoints,
        keypoint_conf=conf,
        camera=camera,
        renderer=r,
        device=device,
        use_progress_bar=verbose
    )

    # if display_result and interactive:
    #     r.wait_for_close()

    return pose, camera_transformation, loss_history, step_imgs


def create_animation(dataset, config, start_idx=0, end_idx=None, device=torch.device('cpu'), dtype=torch.float32, offscreen=False, verbose=False, save_to_file=False):
    final_poses = []

    if end_idx is None:
        end_idx = len(dataset)

    for idx in trange(end_idx - start_idx, desc='Optimizing'):
        idx = start_idx + idx

        final_pose, cam_trans, train_loss, step_imgs = optimize_sample(
            idx,
            dataset,
            config,
            offscreen=offscreen,
            interactive=False
        )

        if verbose:
            print("Optimization of", idx, "frames finished")

        # print("\nPose optimization of frame", idx, "is finished.")
        R = cam_trans.numpy().squeeze()
        idx += 1

        # append optimized pose and camera transformation to the array
        final_poses.append((final_pose, R))

    filename = None

    if save_to_file:
        '''
        Save final_poses array into results folder as a pickle dump
        '''
        results_dir = config['output']['rootDir']
        result_prefix = config['output']['prefix']
        filename = results_dir + get_new_filename()
        print("Saving results to", filename)
        with open(filename, "wb") as fp:
            pickle.dump(final_poses, fp)
        print("Results have been saved to", filename)

    return final_poses, filename
