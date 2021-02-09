# library imports
import os
import pickle
import torch
from tqdm.auto import trange

# local imports
from train_pose import train_pose_with_conf
from model import SMPLyModel
from utils.general import getfilename_from_conf, setup_training
from utils.video import interpolate_poses
from camera_estimation import TorchCameraEstimate


def optimize_sample(sample_index, dataset, config, device=torch.device('cpu'), dtype=torch.float32, interactive=True, offscreen=False, verbose=True, initial_pose=None):
    # prepare data and SMPL model
    model = SMPLyModel.model_from_conf(config, initial_pose=initial_pose)
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

    if not offscreen and interactive:
        # render camera to the scene
        camera.setup_visualization(r.init_keypoints, r.keypoints)

    # train for pose
    best_out, cam_trans, loss_history, step_imgs = train_pose_with_conf(
        config=config,
        model=model,
        keypoints=keypoints,
        keypoint_conf=conf,
        camera=camera,
        renderer=r,
        device=device,
        use_progress_bar=verbose,
        render_steps=(offscreen or interactive)
    )

    # if display_result and interactive:
    #     r.wait_for_close()

    return best_out, cam_trans, loss_history, step_imgs


def create_animation(dataset, config, start_idx=0, end_idx=None, offscreen=False, verbose=False, save_to_file=False, interpolate=False):
    model_outs = []
    use_temporal_data = config['pose']['temporal']['enabled']
    if end_idx is None:
        end_idx = len(dataset) - 1

    initial_pose = None

    for idx in trange(end_idx - start_idx, desc='Optimizing'):
        idx = start_idx + idx

        if use_temporal_data and initial_pose is not None:
            config['pose']['lr'] = config['pose']['temporal']['lr']
            config['pose']['iterations'] = config['pose']['temporal']['iterations']

        best_out, cam_trans, train_loss, step_imgs = optimize_sample(
            idx,
            dataset,
            config,
            verbose=verbose,
            offscreen=offscreen,
            interactive=verbose,
            initial_pose=initial_pose
        )

        if verbose:
            print("Optimization of", idx, "frames finished")

        # print("\nPose optimization of frame", idx, "is finished.")
        R = cam_trans.cpu().numpy().squeeze()
        idx += 1

        if best_out is None:
            print("[error] optimizer produced no pose. Skipping frame:", idx)
            continue

        # append optimized pose and camera transformation to the array
        model_outs.append((best_out, R))

        if use_temporal_data:
            initial_pose = best_out.body_pose.detach().clone().cpu()  # .to(device=device)

    if interpolate:
        model_outs = interpolate_poses(model_outs)

    file_path = None

    if save_to_file:
        '''
        Save final_poses array into results folder as a pickle dump
        '''
        results_dir = config['output']['rootDir']
        result_prefix = config['output']['prefix']

        pkl_name = getfilename_from_conf(config)

        pkl_name = pkl_name + "-" + str(start_idx)
        if end_idx is not None:
            pkl_name = pkl_name + "-" + str(end_idx)

        pkl_name = pkl_name + ".pkl"

        file_path = os.path.join(results_dir, pkl_name)
        print("Saving results to", file_path)
        with open(file_path, "wb") as fp:
            pickle.dump(model_outs, fp)
        print("Results have been saved to", file_path)

    return model_outs, file_path
