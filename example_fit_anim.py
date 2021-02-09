import pickle
import time
from train import create_animation
from utils.video import video_from_pkl
from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 60  # starting index of the frame to optimize for
FINISH_IDX = 70   # choose a big number to optimize for all frames in samples directory
# if False, only run already saved animation without optimization
RUN_OPTIMIZATION = True

result_image = []
idx = START_IDX

device = torch.device('cpu')
dtype = torch.float32

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)


# Rename files in samples directory to uniform format
if config['data']['renameFiles']:
    rename_files(config['data']['rootDir'] + "/")


'''
Optimization part without visualization
'''
if RUN_OPTIMIZATION:
    model_outs, filename = create_animation(
        dataset,
        config,
        START_IDX,
        FINISH_IDX,
        verbose=False,
        offscreen=True,
        save_to_file=True,
        interpolate=True
    )

# TODO: use current body pose and camera transform for next optimization?


def replay_animation(file, start_frame=0, end_frame=None, with_background=False, fps=30, interpolated=False):
    r = Renderer()
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    with open(file, "rb") as fp:
        model_outs = pickle.load(fp)

    if end_frame is None:
        end_frame = len(model_outs)

    for i in range(start_frame, end_frame):
        body_pose = model_outs[i][0]
        camera_transform = model_outs[i][1]

        if with_background:
            # Changing image is too jerky, because the image has to be removed and added each time
            pass
            # img_path = samples_dir + "/" + str(i) + ".png"
            # if r.get_node("image") is not None:
            #     r.remove_node("image")
            # r.render_image_from_path(img_path, name="image", scale=est_scale)

        r.render_model_with_tfs(model_anim, body_pose, keep_pose=True,
                                render_joints=False, transforms=camera_transform, interpolated=interpolated)
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

video_name = getfilename_from_conf(
    config) + "-" + str(START_IDX) + "-" + str(FINISH_IDX)

#video_from_pkl(anim_file, video_name, config)
replay_animation(anim_file, interpolated=True)
