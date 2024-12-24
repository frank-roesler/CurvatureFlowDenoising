import matplotlib.pyplot as plt
import numpy as np


def initPlots(result: np.array, groundTruth: np.array, plot: bool, savePlots: bool) -> tuple:
    if not plot:
        return None
    ratio = result.shape[0] / result.shape[1]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    im0 = ax[0, 0].imshow(result, cmap="viridis")
    im1 = ax[0, 1].imshow(groundTruth, cmap="viridis")
    im2 = ax[0, 2].imshow(result, cmap="viridis")
    im3 = ax[1, 2].imshow(result, cmap="viridis")
    ax[0, 0].title.set_text("noisy image")
    ax[0, 1].title.set_text("ground truth")
    ax[0, 2].title.set_text("model output")
    ax[1, 2].title.set_text("mean curvature thresholds")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 2].axis("off")

    fig.colorbar(im0, ax=ax[0, 0], fraction=ratio * 0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[0, 1], fraction=ratio * 0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[0, 2], fraction=ratio * 0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[1, 2], fraction=ratio * 0.046, pad=0.04, values=[0, 0.5, 1])
    im0.set_clim(0, 1)
    im1.set_clim(0, 1)
    im2.set_clim(0, 1)
    im3.set_clim(0, 1)
    if not savePlots:
        return ax, (im1, im2, im3)

    figSave, axSave = plt.subplots()
    imSave = axSave.imshow(result, cmap="viridis")
    axSave.axis("off")
    imSave.set_clim(0, 1)
    imSave.figure.savefig(f"results/input.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figSave)

    return ax, (im1, im2, im3)


def updatePlots(
    ax,
    ims: tuple,
    result: np.array,
    area: np.array,
    errs: list,
    diffs: list,
    plot: bool,
    it: int,
    freq: int,
    savePlots: bool,
):
    im1, im2, im3 = ims
    if plot and it % freq == 0:
        print(f"Iteration {it}")
        im2.set_data(result)
        im3.set_data(area)
        ax[1, 0].cla()
        ax[1, 1].cla()
        ax[1, 0].plot(diffs, linewidth=0.5)
        ax[1, 1].plot(errs, linewidth=0.5)
        ax[1, 0].title.set_text("Size of PDE update")
        ax[1, 1].title.set_text("L2 error to ground truth")
        plt.show(block=False)
        plt.pause(0.001)

    if not savePlots:
        return

    figSave, axSave = plt.subplots()
    imSave = axSave.imshow(result, cmap="viridis")
    axSave.axis("off")
    imSave.set_clim(0, 1)
    imSave.figure.savefig(f"results/output.png", dpi=300, bbox_inches="tight", pad_inches=0)
    im1.figure.savefig(f"results/info.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figSave)
