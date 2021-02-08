# library imports
from train import optimize_sample
import matplotlib.pyplot as plt

# local imports
from utils.general import load_config
from dataset import SMPLyDataset

# load and select sample
config = load_config()
dataset = SMPLyDataset.from_config(config=config)
sample_index = 0

# train for pose
pose, train_loss, step_imgs = optimize_sample(
    sample_index,
    dataset,
    config
)


# color = r.get_snapshot()
# plt.imshow(color)
# plt.show()

# fig, ax = plt.subplots()
# name = getfilename_from_conf(config=config, index=sample_index)
# ax.plot(train_loss[1::], label='sgd')
# ax.set(xlabel="Training iteration", ylabel="Loss", title='Training loss')
# fig.savefig("results/" + name + ".png")
# ax.legend()
# plt.show()
