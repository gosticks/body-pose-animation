import pickle
import time
from train import create_animation
from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 1  # starting index of the frame to optimize for
FINISH_IDX = 200   # choose a big number to optimize for all frames in samples directory
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
        interpolate=False
    )


def replay_animation(file, start_frame=0, end_frame=None, with_background=False, fps=30, interpolated=False):
    r = Renderer()
    r.start()

    model_anim = SMPLyModel.model_from_conf(config)

    with open(file, "rb") as fp:
        results = pickle.load(fp)

    if end_frame is None:
        end_frame = len(results)

    for model, camera_transform in results[start_frame::]:
        if interpolated:
            vertices = model
        else:
            vertices = model.vertices

        r.render_model_geometry(
            faces=model_anim.faces,
            vertices=vertices,
            pose=camera_transform
        )

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

replay_animation(anim_file, interpolated=True)
