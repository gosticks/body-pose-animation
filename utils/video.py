import pickle
from model import SMPLyModel
from renderer import DefaultRenderer
import cv2
from tqdm import tqdm
import numpy as np
from scipy import interpolate

def make_video(images, video_name: str, fps=5, ext: str = "mp4"):
    images = np.array(images)
    width = images.shape[2]
    height = images.shape[1]

    fourcc = 0
    if ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    video_name = video_name + "." + ext

    video = cv2.VideoWriter(
        video_name, fourcc, fps, (width, height), True)

    print("creating video with size", width, height)

    for idx in tqdm(range(len(images))):
        img = images[idx]
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(im_rgb)

    video.release()


def video_from_pkl(filename, video_name, config, ext: str = "mp4"):
    with open(filename, "rb") as fp:
        model_outs = pickle.load(fp)
    save_to_video(model_outs, video_name, config)


def save_to_video(poses, video_name, config, fps=30, interpolated=False):
    r = DefaultRenderer(
        offscreen=True
    )
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    frames = []

    for body_pose, cam_trans in tqdm(poses):
        r.render_model_with_tfs(model_anim, body_pose, keep_pose=True,
                                render_joints=False, transforms=cam_trans, interpolated=interpolated)
        frames.append(r.get_snapshot())

    make_video(frames, video_name, fps)

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

            evenly_spaced_points = np.linspace(x[0], x[-1], (poses_pair.shape[0] - 1) * (num_intermediate + 1) + 1)

            new_frames = f1(evenly_spaced_points)
            new_cameras = f2(evenly_spaced_points)

            arr = [(new_frames[i], new_cameras[i]) for i in range(new_frames.shape[0])]
            if 0 < i < len(poses) - 1:
                arr.pop(0)  # remove first frame that was already added in the last interpolation
            new_poses += arr

    return new_poses
