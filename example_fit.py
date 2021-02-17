# library imports
import math
import os
from train import optimize_sample
import matplotlib.pyplot as plt

# local imports
from utils.general import load_config
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


# color = r.get_snapshot()
# plt.imshow(color)
# plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_history[1::], label='sgd')
ax[0].set(xlabel="Iterations", ylabel="Loss", title='Total Loss')
plt_idx = 1
for name, loss in loss_components.items():
    x = math.floor(plt_idx / 3)
    y = plt_idx % 2
    ax[1].plot(loss[1::], label=name)
    ax[1].set(xlabel="Iteration",
              ylabel="Loss", title="Component Loss")

    plt_idx = plt_idx + 1

plt.legend(loc="upper left")
# name = getfilename_from_conf(config=config, index=sample_index)
# fig.savefig("results/" + name + ".png")
# ax.legend()
plt.show()
