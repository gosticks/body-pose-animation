import os
from utils.video import make_video, make_video_with_pip
from utils.graphs import render_loss_graph
from train import optimize_sample

# local imports
from utils.general import get_output_path_from_conf, load_config
from dataset import SMPLyDataset

# load and select sample
config = load_config()
dataset = SMPLyDataset.from_config(config=config)
sample_index = 150
save_to_video = True


if os.getenv('SAMPLE_INDEX') is not None:
    sample_index = int(os.getenv('SAMPLE_INDEX'))


# train for pose
pose, camera_transformation, loss_history, step_imgs, loss_components = optimize_sample(
    sample_index,
    dataset,
    config,
    interactive=not save_to_video,
    offscreen=save_to_video
)

if save_to_video:
    img_path = dataset.get_image_path(sample_index)
    make_video_with_pip(step_imgs, pip_image_path=img_path,
                        video_name="example_fit")


filename = get_output_path_from_conf(config) + ".png"
render_loss_graph(
    loss_history=loss_history,
    loss_components=loss_components,
    save=True,
    filename=filename)
