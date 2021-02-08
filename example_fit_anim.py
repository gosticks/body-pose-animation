import pickle
import time
from train import create_animation
from tqdm import tqdm
from utils.video import make_video, video_from_pkl
import torch

from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 0  # starting index of the frame to optimize for
FINISH_IDX = 2   # choose a big number to optimize for all frames in samples directory
# if False, only run already saved animation without optimization
RUN_OPTIMIZATION = False

result_image = []
idx = START_IDX

device = torch.device('cpu')
dtype = torch.float32

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)


def get_next_frame(idx):
    """
    Get keypoints and image_path of the frame given index.

    :param idx: index of the frame
    :return: tuple of keypoints, conf and image path
    """
    keypoints, keypoints_conf = dataset[idx]
    if keypoints is None:
        return
    image_path = dataset.get_image_path(idx)
    return keypoints, keypoints_conf, image_path


# Rename files in samples directory to uniform format
if config['data']['renameFiles']:
    rename_files(config['data']['rootDir'] + "/")


'''
Optimization part without visualization
'''
if RUN_OPTIMIZATION:
    final_poses, filename = create_animation(
        dataset,
        config,
        START_IDX,
        FINISH_IDX,
        offscreen=True,
        save_to_file=True
    )


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

# TODO: use current body pose and camera transform for next optimization?


def replay_animation(file, start_frame=0, end_frame=None, with_background=False, fps=30):
    r = Renderer()
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    with open(file, "rb") as fp:
        final_poses = pickle.load(fp)

    if end_frame is None:
        end_frame = len(final_poses)

    for i in range(start_frame, end_frame):
        body_pose = final_poses[i][0]
        camera_transform = final_poses[i][1]

        if with_background:
            # Changing image is too jerky, because the image has to be removed and added each time
            pass
            # img_path = samples_dir + "/" + str(i) + ".png"
            # if r.get_node("image") is not None:
            #     r.remove_node("image")
            # r.render_image_from_path(img_path, name="image", scale=est_scale)

        r.render_model_with_tfs(model_anim, body_pose, keep_pose=True,
                                render_joints=False, transforms=camera_transform)
        time.sleep(1 / fps)


'''
Play the animation.
'''
if RUN_OPTIMIZATION:
    anim_file = filename
else:
    results_dir = config['output']['rootDir']
    result_prefix = config['output']['prefix']
    anim_file = results_dir + result_prefix + "0.pkl"

video_from_pkl(anim_file, "test-anim.avi", config)
replay_animation(anim_file)
