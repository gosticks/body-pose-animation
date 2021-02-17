import matplotlib.pyplot as plt
import math


def render_loss_graph(
        loss_history,
        loss_components,
        save=False,
        show=True,
        filename="untitled.png"):
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

    if save:
        fig.savefig(filename)
    if show:
        plt.show()
