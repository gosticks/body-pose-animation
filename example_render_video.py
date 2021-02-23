import pickle
import time
from train import create_animation
from utils.video import save_to_video
from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 150  # starting index of the frame to optimize for
FINISH_IDX = 300  # choose a big number to optimize for all frames in samples directory

result_image = []
idx = START_IDX

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)

model_outs, filename = create_animation(
    dataset,
    config,
    START_IDX,
    FINISH_IDX,
    verbose=False,
    offscreen=False,
    save_to_file=False,
    interpolate=False
)

video_name = get_output_path_from_conf(
    config) + "-" + str(START_IDX) + "-" + str(FINISH_IDX)

save_to_video(model_outs, video_name, config,
              start_frame_offset=START_IDX,
              dataset=dataset)
