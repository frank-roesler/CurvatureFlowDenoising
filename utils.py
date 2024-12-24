import matplotlib.pyplot as plt
import numpy as np


def initPlots(result: np.array, gtoundTruth: np.array, plot: bool, savePlots: bool) -> tuple:
    if not plot:
        return None
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    im0 = ax[0, 0].imshow(result, cmap="viridis")
    im1 = ax[0, 1].imshow(gtoundTruth, cmap="viridis")
    im2 = ax[0, 2].imshow(result, cmap="viridis")
    im3 = ax[1, 2].imshow(result, cmap="viridis")
    im0.title.set_text("noisy image")
    im1.title.set_text("ground truth")
    im2.title.set_text("model output")
    im3.title.set_text("mean curvature")

    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 2].axis("off")

    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[1, 2], fraction=0.046, pad=0.04)
    im0.set_clim(0, 1)
    im1.set_clim(0, 1)
    im2.set_clim(0, 1)
    im3.set_clim(0, 1)
    if not savePlots:
        return ax, im1, im2, im3

    figSave, axSave = plt.subplots()
    imSave = axSave.imshow(result, cmap="viridis")
    axSave.axis("off")
    imSave.set_clim(0, 1)
    imSave.figure.savefig(f"results/input.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figSave)

    return ax, im1, im2, im3


def updatePlots(
    ax,
    im1,
    im2,
    im3,
    result: np.array,
    area: np.array,
    errs: list,
    diffs: list,
    plot: bool,
    it: int,
    freq: int,
    savePlots: bool,
):
    if plot and it % freq == 0:
        print(f"Iteration {it}")
        ax[1, 0].cla()
        ax[1, 1].cla()
        im2.set_data(result)
        im3.set_data(area)
        ax[1, 0].semilogy(diffs, linewidth=0.5)
        ax[1, 1].semilogy(errs, linewidth=0.5)
        plt.show(block=False)
        plt.pause(0.001)

    if not savePlots:
        return

    figSave, axSave = plt.subplots()
    imSave = axSave.imshow(result, cmap="viridis")
    axSave.axis("off")
    imSave.set_clim(0, 1)
    imSave.figure.savefig(f"results/output.png", dpi=300, bbox_inches="tight", pad_inches=0)
    im1.figure.savefig(f"results/thresholds.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figSave)
