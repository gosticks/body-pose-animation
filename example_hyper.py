import pickle
import time
from train import create_animation
from tqdm import tqdm
from utils.video import make_video, save_to_video, video_from_pkl
import torch
import itertools
import numpy as np
from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from utils.general import rename_files, get_new_filename

START_IDX = 00  # starting index of the frame to optimize for
FINISH_IDX = 20   # choose a big number to optimize for all frames in samples

device = torch.device('cpu')
dtype = torch.float32

config = load_config()
dataset = SMPLyDataset.from_config(config)
model = SMPLyModel.model_from_conf(config)


def run_test(config):
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
    video_name = getfilename_from_conf(
        config) + "-" + str(START_IDX) + "-" + str(FINISH_IDX)

    video_name = os.path.join(config['output']['rootDir'], video_name)

    video_name = getfilename_from_conf(
        config) + "-" + str(START_IDX) + "-" + str(FINISH_IDX)

    video_name = os.path.join(
        config['output']['rootDir'],
        video_name
    )

    save_to_video(
        model_outs, video_name, config,
        dataset=dataset, interpolation_target=60
    )


def run_pose_tests(config):
    priors_types = ['bodyPrior', 'anglePrior', 'angleSumLoss', 'temporal']
    l = [False, True]
    permutations = [list(i)
                    for i in itertools.product(l, repeat=len(priors_types))]

    lr_steps = [0.01, 0.02, 0.03, 0.04, 0.05,
                0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for lr in lr_steps:
        print("running test with lr:", lr)
        config['pose']['lr'] = lr

        for p in permutations:
            print("running test:", config['pose']['optimizer'])
            # iterate over all permutations and update config
            for i, v in enumerate(p):
                config['pose'][priors_types[i]]['enabled'] = v
                print(priors_types[i] + ":", v)
            run_test(config)
            print("------------------------------------------------")


print("training: Adam")
# run tests for adam
run_pose_tests(config)

print("training: LBFGS")
# try the same with lbfgs
config = load_config("./config.lbfgs.temporal.yaml")
run_pose_tests(config)
