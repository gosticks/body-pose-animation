import pickle
from model import SMPLyModel
from renderer import DefaultRenderer
import cv2
from tqdm import tqdm
import numpy as np


def make_video(images, video_name: str, fps=5):
    images = np.array(images)
    width = images.shape[2]
    height = images.shape[1]
    video = cv2.VideoWriter(
        video_name, 0, fps, (width, height), True)

    print("creating video with size", width, height)

    for idx in tqdm(range(len(images))):
        img = images[idx]
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(im_rgb)

    video.release()


def video_from_pkl(filename, video_name, config):
    with open(filename, "rb") as fp:
        final_poses = pickle.load(fp)

    save_to_video(final_poses, video_name, config)


def save_to_video(poses, video_name, config, fps=30):
    r = DefaultRenderer(
        offscreen=True
    )
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    frames = []

    for body_pose, cam_trans in tqdm(poses):
        r.render_model_with_tfs(model_anim, body_pose, keep_pose=True,
                                render_joints=False, transforms=cam_trans)
        frames.append(r.get_snapshot())

    make_video(frames, video_name, fps)
