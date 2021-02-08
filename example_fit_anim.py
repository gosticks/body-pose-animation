import pickle
import time
from utils.render import make_video
import torch
from tqdm.std import tqdm

from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from camera_estimation import TorchCameraEstimate
from modules.camera import SimpleCamera
from train_pose import train_pose_with_conf
from utils.general import rename_files, get_new_filename

START_IDX = 150  # starting index of the frame to optimize for
FINISH_IDX = 200   # choose a big number to optimize for all frames in samples directory
# if False, only run already saved animation without optimization
RUN_OPTIMIZATION = False

final_poses = []  # optimized poses array that is saved for playing the animation
result_image = []
idx = START_IDX


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


device = torch.device('cpu')
dtype = torch.float32

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)

samples_dir = config['data']['rootDir']

# Rename files in samples directory to uniform format
if config['data']['renameFiles']:
    rename_files(samples_dir + "/")

results_dir = config['output']['rootDir']
result_prefix = config['output']['prefix']

model_out = model()
joints = model_out.joints.detach().cpu().numpy().squeeze()

'''
Optimization part without visualization
'''
if RUN_OPTIMIZATION:
    for idx in dataset:

        init_keypoints, init_joints, keypoints, conf, est_scale, r, img_path = setup_training(
            model=model,
            renderer=True,
            offscreen=True,
            dataset=dataset,
            sample_index=idx
        )

        camera = TorchCameraEstimate(
            model,
            dataset=dataset,
            keypoints=keypoints,
            renderer=None,
            device=torch.device('cpu'),
            dtype=torch.float32,
            image_path=img_path,
            est_scale=est_scale
        )

        print("\nCamera optimization of frame", idx, "is finished.")
        camera = SimpleCamera.from_estimation_cam(camera)

        final_pose = train_pose_with_conf(
            config=config,
            model=model,
            keypoints=keypoints,
            keypoint_conf=conf,
            camera=camera,
            renderer=r,
            device=device
        )

        print("\nPose optimization of frame", idx, "is finished.")
        R = camera.trans.numpy().squeeze()
        idx += 1

        # append optimized pose and camera transformation to the array
        final_poses.append((final_pose, R))

    print("Optimization of", idx, "frames finished")

    '''
    Save final_poses array into results folder as a pickle dump
    '''
    filename = results_dir + get_new_filename()
    print("Saving results to", filename)
    with open(filename, "wb") as fp:
        pickle.dump(final_poses, fp)
    print("Results have been saved to", filename)

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


def video_from_pkl(filename, video_name):
    with open(filename, "rb") as fp:
        final_poses = pickle.load(fp)

    save_to_video(final_poses, video_name)


def save_to_video(poses, video_name, fps=30):
    r = DefaultRenderer(
        offscreen=True
    )
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    frames = []

    for body_pose, cam_trans in tqdm(poses):
        r.render_model_with_tfs(model_anim, body_pose, keep_pose=True,
                                render_joints=False, transforms=cam_trans)
        frames.append(r.get_snapshot(False))

    make_video(frames, video_name, fps)


'''
Play the animation.
'''
anim_file = results_dir + result_prefix + "0.pkl"
if RUN_OPTIMIZATION:
    anim_file = filename

replay_animation(anim_file)
