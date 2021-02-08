import pickle
import time
from tqdm import tqdm
from utils.render import make_video
import torch
from tqdm.auto import trange

from dataset import SMPLyDataset
from model import *
from utils.general import *
from renderer import *
from camera_estimation import TorchCameraEstimate
from modules.camera import SimpleCamera
from train_pose import train_pose_with_conf
from utils.general import rename_files, get_new_filename


def animate_with_conf(config, start=0, end=None):
