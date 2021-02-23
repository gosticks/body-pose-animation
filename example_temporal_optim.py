import pickle
import time
from train import create_animation
from tqdm import tqdm
from utils.video import make_video, video_from_pkl
import torch
import itertools
import numpy as np
from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 50  # starting index of the frame to optimize for
FINISH_IDX = 100   # choose a big number to optimize for all frames in samples

device = torch.device('cpu')
dtype = torch.float32

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)


def run_test(config):
    _, filename = create_animation(
        dataset,
        config,
        START_IDX,
        FINISH_IDX,
        verbose=False,
        offscreen=False,
        save_to_file=True
    )
    video_name = getfilename_from_conf(
        config) + "-" + str(START_IDX) + "-" + str(FINISH_IDX)

    video_name = os.path.join(config['output']['rootDir'], video_name)

    video_from_pkl(filename, video_name, config)


def run_pose_tests(config):
    priors_types = ['temporal']
    l = [True, False]
    permutations = [list(i)
                    for i in itertools.product(l, repeat=len(priors_types))]

    for p in permutations:
        print("running test:", config['pose']['optimizer'])
        # iterate over all permutations and update config
        for i, v in enumerate(p):
            config['pose'][priors_types[i]]['enabled'] = v
            print(priors_types[i] + ":", v)
        run_test(config)
        print("------------------------------------------------")


print("training: LBFGS")
# run tests for adam
run_pose_tests(config)

# print("training: LBFGS")
# # try the same with lbfgs
# config = load_config("./config.lbfgs.temporal.yaml")
# run_pose_tests(config)
