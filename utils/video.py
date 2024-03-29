from dataset import SMPLyDataset
import pickle
from typing import Tuple
from model import SMPLyModel
from renderer import DefaultRenderer
import cv2
from tqdm import tqdm
import numpy as np
from scipy import interpolate


def make_video(images, video_name: str, fps=30, ext: str = "mp4", post_process_frame=None):
    images = np.array(images)
    width = images.shape[2]
    height = images.shape[1]

    fourcc = 0
    if ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    video_name = video_name + "." + ext

    video = cv2.VideoWriter(
        video_name, fourcc, fps, (width, height), True)

    for idx in tqdm(range(len(images))):
        img = images[idx]
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if post_process_frame is not None:
            img_rgb = post_process_frame(img=im_rgb, idx=idx)

        video.write(im_rgb)

    video.release()
    print("video saved to:", video_name)


def video_from_pkl(filename, video_name, config, ext: str = "mp4"):
    with open(filename, "rb") as fp:
        model_outs = pickle.load(fp)
    save_to_video(model_outs, video_name, config)


def save_to_video(
    sample_output: Tuple,
    video_name: str,
    config: object,
    fps=30,
    include_thumbnail=True,
    thumbnail_size=0.2,
    start_frame_offset=0,
    dataset: SMPLyDataset = None,
    interpolation_target=None
):
    """
    Renders a video from pose, camera tuples. Additionally interpolation can be used to smooth out the animation

    Args:
        sample_output (Tuple): A tuple of body pose vertices and a camera transformation
        video_name (str): name for the resulting video file (can also be a path)
        config (object): general run config
        fps (int, optional): animation base fps. Defaults to 30.
        interpolation_target (int, optional): expand animation fps via interpolation to this target. Defaults to 60.
    """
    r = DefaultRenderer(
        offscreen=True
    )
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    if interpolation_target is not None:
        if interpolation_target % fps != 0:
            print("[error] interpolation target must be a multiple of fps")
            return
        inter_ratio = int(interpolation_target / fps)
        num_intermediate = inter_ratio - 1
        sample_output = interpolate_poses(sample_output, num_intermediate)
    else:
        sample_output = [
            (
                out.vertices.detach().cpu().numpy()[0],
                cam
            ) for out, cam in sample_output]
    frames = []
    print("[export] rendering animation frames...", sample_output[0][0].shape)

    # just use the first transform
    cam_transform = sample_output[0][1]

    for vertices, cam_trans in tqdm(sample_output):
        r.render_model_geometry(
            faces=model_anim.faces,
            vertices=vertices,
            pose=cam_trans  # cam_transform,
        )
        frames.append(r.get_snapshot())

    target_fps = fps
    if interpolation_target is not None:
        target_fps = interpolation_target

    def post_process_frame(img, idx: int):
        if not include_thumbnail:
            return img
        # account for start from frames not zero
        idx = start_frame_offset + idx
        frame_idx = idx
        if interpolation_target is not None:
            # account for possible interpolation
            frame_idx = int(idx / inter_ratio)
        img_path = dataset.get_image_path(frame_idx)
        overlay = cv2.imread(img_path)

        if overlay is None:
            print("[error] image could not be ", img_path)
            return img

        overlay = cv2.resize(
            overlay,
            dsize=(
                int(overlay.shape[1] * thumbnail_size),
                int(overlay.shape[0] * thumbnail_size)
            ))
        img[0:overlay.shape[0], 0:overlay.shape[1]] = overlay
        return img

    make_video(frames, video_name, target_fps,
               post_process_frame=post_process_frame)


def make_video_with_pip(frames, pip_image_path, video_name: str, fps=30, ext: str = "mp4", image_size=0.2):
    """renders a video with a pip frame in the corner
    """

    def post_process_frame(img, idx: int):
        overlay = cv2.imread(pip_image_path)

        if overlay is None:
            print("[error] image could not be ", pip_image_path)
            return img

        overlay = cv2.resize(
            overlay,
            dsize=(
                int(overlay.shape[1] * image_size),
                int(overlay.shape[0] * image_size)
            ))
        img[0:overlay.shape[0], 0:overlay.shape[1]] = overlay
        return img

    make_video(frames, video_name, fps,
               post_process_frame=post_process_frame)


def interpolate_poses(poses, num_intermediate=5):
    """
    Interpolate vertices and cameras between pairs of frames by adding intermediate results

    :param poses: optimized poses
    :param num_intermediate: amount of intermediate results to insert between each pair of frames
    :return: interpolated poses, list of tuples (body_pose, camera_pose)
    """
    new_poses = []
    for i in range(len(poses) - 1):
        if len(poses) < 2:
            return poses
        else:
            # Shape of one matrix of vertices = torch.Size([1, 10475, 3])
            pose_1 = poses[i][0].vertices.detach().cpu().numpy()
            pose_2 = poses[i + 1][0].vertices.detach().cpu().numpy()
            poses_pair = np.concatenate((pose_1, pose_2), axis=0)

            camera_1 = np.expand_dims(poses[i][1], axis=0)
            camera_2 = np.expand_dims(poses[i + 1][1], axis=0)
            camera_pair = np.concatenate((camera_1, camera_2), axis=0)

            x = np.arange(poses_pair.shape[0])
            f1 = interpolate.interp1d(x, poses_pair, axis=0)
            f2 = interpolate.interp1d(x, camera_pair, axis=0)

            evenly_spaced_points = np.linspace(
                x[0], x[-1], (poses_pair.shape[0] - 1) * (num_intermediate + 1) + 1)

            new_frames = f1(evenly_spaced_points)
            new_cameras = f2(evenly_spaced_points)

            arr = [(new_frames[i], new_cameras[i])
                   for i in range(new_frames.shape[0])]
            if 0 < i < len(poses) - 1:
                # remove first frame that was already added in the last interpolation
                arr.pop(0)
            new_poses += arr

    return new_poses
