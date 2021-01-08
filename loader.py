import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import os
import json
import numpy as nn

class DataLoader():
    def __init__(self, data_path):
        print("DataLoader: input location" + data_path)
        self.data_path = data_path

    # display current sample that is loaded
    # only used for debug while developing
    def show_cur_item(self):
        img = mpimg.imread(os.path.join(self.data_path, 'frame-070.jpg'))
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img)
        self.draw_keypoints(ax)
        plt.show()

    # load people data from a openpose json dump
    def load_people(self):
        path = os.path.join(self.data_path, 'input_000000000068_keypoints.json')
        with open(path) as file:
            json_data = json.load(file)
            people = json_data['people']
        return people

    # render keypoints on top of current image
    def draw_keypoints(self, plot):
        for p in self.load_people():
            keypoints = nn.array(p['pose_keypoints_2d']).reshape(-1, 3)

            for point in keypoints:
                # draw points in image
                marker = Circle((point[0], point[1]), 5 * point[2])
                plot.add_patch(marker)
        print("drawing keypoints")
