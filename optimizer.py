import torch
import torch.nn as nn


class CameraLoss(nn.Module):
    def __init__(self):
        print("CameraLoss init")

    def forward(self, keypoints, joints, transform):

        # simple stupid implementation
        # compute 2d distance between model and openpose joints
        # ignoring Z axis
        joints * transform
