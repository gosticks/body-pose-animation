import matplotlib.pyplot as plt
import math


def name_for_loss(name):
    if name == "points":
        return "MSE loss"
    if name == "BodyPrior":
        return "body prior"
    if name == "AngleSumLoss":
        return "angle sum loss"
    if name == "AnglePriorsLoss":
        return "angle prior"
    if name == "AngleClipper":
        return "angle threshold loss"
    if name == "IntersectLoss":
        return "mesh intersection loss"

    return name


def color_for_loss(name):
    if name == "points":
        return "C6"
    if name == "BodyPrior":
        return "C2"
    if name == "AnglePriorsLoss":
        return "C3"
    if name == "AngleClipper":
        return "C4"
    if name == "IntersectLoss":
        return "C5"
    if name == "AngleSumLoss":
        return "C1"

    return None


def render_loss_graph(
        loss_history,
        loss_components,
        save=False,
        show=True,
        filename="untitled.png"):

    fig = plt.figure(figsize=(8, 5))

    ax = fig.subplots(1, 2)
    ax[0].plot(loss_history[1::], label='sgd')
    ax[0].set(xlabel="Iterations", ylabel="Loss", title='Total Loss')
    plt_idx = 1
    for name, loss in loss_components.items():
        x = math.floor(plt_idx / 3)
        y = plt_idx % 2
        ax[1].plot(loss[1::], label=name_for_loss(
            name), color=color_for_loss(name))
        ax[1].set(xlabel="Iteration",
                  ylabel="Loss", title="Component Loss")

        plt_idx = plt_idx + 1

    plt.legend(loc="best")

    if save:
        fig.savefig(filename)
    if show:
        plt.show()
