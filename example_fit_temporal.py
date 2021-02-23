import os
from utils.video import make_video_with_pip
from utils.graphs import render_loss_graph
from train import optimize_sample

# local imports
from utils.general import get_output_path_from_conf, load_config
from dataset import SMPLyDataset

# load and select sample
config = load_config()
dataset = SMPLyDataset.from_config(config=config)
sample_index = 55
save_to_video = True

if os.getenv('SAMPLE_INDEX') is not None:
    sample_index = int(os.getenv('SAMPLE_INDEX'))

# train for pose
best_pose, camera_transformation, loss_history, step_imgs, loss_components = optimize_sample(
    sample_index - 1,
    dataset,
    config,
    interactive=False
)

config['pose']['lr'] = config['pose']['temporal']['lr']
config['pose']['iterations'] = config['pose']['temporal']['iterations']

# reuse model to train new sample
pose_temp, camera_transformation, loss_history, step_imgs, loss_components = optimize_sample(
    sample_index,
    dataset,
    config,
    interactive=False,
    offscreen=True,
    initial_pose=best_pose.body_pose.detach().clone().cpu(),
    initial_orient=best_pose.global_orient.detach().clone().cpu()
)

if save_to_video:
    img_path = dataset.get_image_path(sample_index)
    make_video_with_pip(step_imgs, pip_image_path=img_path,
                        video_name="example_fit_temporal")

filename = get_output_path_from_conf(config) + ".png"
render_loss_graph(
    loss_history=loss_history,
    loss_components=loss_components,
    save=True,
    filename=filename)
