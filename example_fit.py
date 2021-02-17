import os
from utils.graphs import render_loss_graph
from train import optimize_sample

# local imports
from utils.general import get_output_path_from_conf, load_config
from dataset import SMPLyDataset

# load and select sample
config = load_config()
dataset = SMPLyDataset.from_config(config=config)
sample_index = 55

if os.getenv('SAMPLE_INDEX') is not None:
    sample_index = int(os.getenv('SAMPLE_INDEX'))

# train for pose
pose, camera_transformation, loss_history, step_imgs, loss_components = optimize_sample(
    sample_index,
    dataset,
    config,
    interactive=True
)


filename = get_output_path_from_conf(config) + ".png"
render_loss_graph(
    loss_history=loss_history,
    loss_components=loss_components,
    save=True,
    filename=filename)
