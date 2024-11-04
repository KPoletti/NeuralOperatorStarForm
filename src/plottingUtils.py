"""This module contains functions for plotting the data. 
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_style("white")


def plot_timeStep_NoBounds(
    truthData, predcData, error, timeStep, save=False, saveName=""
):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 8)
    grid = 1.25

    im1 = ax[0].imshow(
        truthData[timeStep, :, :],
        cmap="viridis",
        # norm=matplotlib.colors.LogNorm(),
        origin="lower",
        extent=[-grid, grid, -grid, grid],
    )
    im2 = ax[1].imshow(
        predcData[timeStep, :, :],
        cmap="viridis",
        # norm=matplotlib.colors.LogNorm(),
        origin="lower",
        extent=[-grid, grid, -grid, grid],
    )
    im3 = ax[2].imshow(
        error[timeStep, :, :],
        cmap="viridis",
        # norm=matplotlib.colors.LogNorm(),
        origin="lower",
        extent=[-grid, grid, -grid, grid],
    )
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        im1,
        cax=cax0,
        orientation="vertical",
        label=r"Projected Density ($\frac{g}{cm^2}$)",
    )
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        im1,
        cax=cax1,
        orientation="vertical",
        label=r"Projected Density ($\frac{g}{cm^2}$)",
    )
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        im3, cax=cax2, orientation="vertical", label=r"Error ($\frac{g}{cm^2}$)"
    )
    fig.tight_layout()

    # fig.colorbar(im4, ax=ax[1, 0], shrink=0.57, label=r"Error ($\frac{g}{cm^2}$)")
    if save:
        fig.savefig(f"{saveName}.png")
    # plt.show()
    return fig, ax, im1, im2, im3


def Animation_true_pred_error(
    truthData,
    predcData,
    savePath,
    out_times,
    mass,
    error_type="rel",
    fps=25,
    device=0,
):
    if device != 0:
        return None
    truthData = truthData.cpu().detach().numpy()
    predcData = predcData.cpu().detach().numpy()
    numOfFrames = np.min([500, truthData.shape[0]])
    error_type = "absErr"
    log_axis = ""
    error2use = np.abs(truthData - predcData)
    error_title = r"Absolute Error  $|\rho_{pred} - \rho_{true}|$"

    if "rel" in error_type:
        error2use = np.abs(truthData - predcData) / np.abs(truthData)
        error_title = (
            r"Relative Error  $\frac{|\rho_{pred} - \rho_{truth}|}{\|\rho_{truth}\|}$"
        )

    movie_name = f"{savePath}FlowVideo_{error_type}{log_axis}.mp4"
    fig, ax, im1, im2, im3 = plot_timeStep_NoBounds(
        truthData, predcData, error2use, 0, save=False, saveName="test"
    )
    im1.set_clim(vmin=truthData.min(), vmax=truthData.mean() + 6 * truthData.std())
    im2.set_clim(vmin=truthData.min(), vmax=truthData.mean() + 6 * truthData.std())
    im3.set_clim(
        vmin=error2use.min() + 1e-3, vmax=error2use.mean() + 5 * error2use.std()
    )

    # fig.set_size_inches(20,20)

    # ax[0, 0].set_title("Input Data at t")
    ax[0].set_title(r"True Density")
    ax[1].set_title(r"FNO Predicted Density")
    ax[2].set_title(error_title)

    # ax[0].set_title("MHD Data")
    ax[0].set_xlabel("Length (pc)")
    ax[1].set_xlabel("Length (pc)")
    ax[2].set_xlabel("Length (pc)")
    ax[0].set_ylabel("Length (pc)")

    # fig.tight_layout()
    def animate(i):
        # im1.set_data(inputData[i, ...])
        im1.set_data(truthData[i, ...])
        im2.set_data(predcData[i, ...])
        im3.set_data(error2use[i, ...])
        if i % 5 == 0:
            fig.suptitle(f"T = {out_times[i] *10**-3:1.3f} Myr")
            if "roll" in savePath:
                im1.set_clim(
                    vmin=truthData[i, ...].min(),
                    vmax=truthData.mean() + 4 * truthData.std(),
                )
                im2.set_clim(
                    vmin=predcData[i, ...].min(),
                    vmax=predcData.mean() + 4 * predcData.std(),
                )
                im3.set_clim(
                    vmin=error2use[i, ...].min(),
                    vmax=error2use.mean() + 4 * error2use.std(),
                )

    ani = animation.FuncAnimation(
        fig, animate, frames=numOfFrames, interval=100, repeat=True
    )
    ani.save(
        movie_name,
        dpi=100,
        writer=animation.FFMpegWriter(fps=fps),
    )

    print(f"Saved to {movie_name}")
    plt.close()


def createAnimation(
    data: torch.Tensor, time: torch.Tensor, save_dir: str, mass, fps=10
):
    """
    Creates an animation of the density and velocity data.

    Args:
        data (torch.Tensor): The data to animate.
        time (torch.Tensor): The time data.
        save_dir (str): The directory to save the animation.
        mass: The mass data.
        fps (int, optional): The frames per second of the animation. Defaults to 10.

    Returns:
        None
    """
    if isinstance(mass, str):
        mass = [mass]
    # create an animation of the density and velocity data
    tmp = data.cpu().detach().numpy()
    fig, axe = plt.subplots(1, 3, figsize=(10, 5))
    im1 = axe[0].imshow(tmp[0, ..., 0], cmap="inferno")
    axe[0].set_title(f"Density Mass {mass[0]} M-sun Timestep {float(time[0]):0.3f} kyr")
    im2 = axe[1].imshow(tmp[0, ..., 1], cmap="inferno")
    axe[1].set_title("Velocity X")
    im3 = axe[2].imshow(tmp[0, ..., 2], cmap="inferno")
    axe[2].set_title("Velocity Y")
    im1.set_clim(vmin=tmp[0, ..., 0].min(), vmax=tmp[0, ..., 0].max())
    im2.set_clim(vmin=tmp[0, ..., 1].min(), vmax=tmp[0, ..., 1].max())
    im3.set_clim(vmin=tmp[0, ..., 2].min(), vmax=tmp[0, ..., 2].max())

    # add colorbar to each axis
    fig.colorbar(
        im1,
        ax=axe[0],
        shrink=0.4,
        label=r"Projected Density ($\frac{g}{cm^2}$)",
    )
    fig.colorbar(
        im3,
        ax=axe[2],
        shrink=0.4,
        label=r"Velocity X ($\frac{cm}{s}$)",
    )
    fig.colorbar(
        im2,
        ax=axe[1],
        shrink=0.4,
        label=r"Velocity Y ($\frac{cm}{s}$)",
    )

    plt.tight_layout()

    num_frames = data.shape[0]

    def animate(i):
        im1.set_data(tmp[i, ..., 0])
        im2.set_data(tmp[i, ..., 1])
        im3.set_data(tmp[i, ..., 2])
        if i % 2 == 0:
            im1.set_clim(vmin=tmp[i, ..., 0].min(), vmax=tmp[i, ..., 0].max())
            im2.set_clim(vmin=tmp[i, ..., 1].min(), vmax=tmp[i, ..., 1].max())
            im3.set_clim(vmin=tmp[i, ..., 2].min(), vmax=tmp[i, ..., 2].max())
        if len(mass) > 1:
            axe[0].set_title(
                f"Density Mass {mass[i]} Timestep {float(time[i]):0.3f} kyr"
            )
        else:
            axe[0].set_title(
                f"Density Mass {mass[0]} Timestep {float(time[i]):0.3f} kyr"
            )

    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    ani.save(
        f"{save_dir}_M{mass[0]}_Movie.mp4", fps=fps, extra_args=["-vcodec", "libx264"]
    )
    plt.close()


def randomPlot(
    indata: torch.Tensor, output: torch.Tensor, target: torch.Tensor, savename: str
):
    """
    Plots a time point of the input, output, and target.

    Args:
        input (torch.Tensor): The input data.
        output (torch.Tensor): The output data.
        target (torch.Tensor): The target data.
        savename (str): The name to save the plot.

    Returns:
        None
    """
    # select random first and second dimension
    idx1 = np.random.randint(0, indata.shape[0])
    idx2 = np.random.randint(0, indata.shape[1])
    # check the shape of the input
    dim = "1 D problem"
    if len(indata.shape) > 4:
        dim = np.random.randint(0, indata.shape[-1])
        dim = 0
        indata = indata[..., dim]
        output = output[..., dim]
        target = target[..., dim]

    # create 1 by 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plot the input
    im0 = axs[0].imshow(indata[idx1, idx2, :, :].cpu().detach().numpy())
    axs[0].set_title(f"Input: Timestep {idx1}, {idx2}. Dim {dim}")
    fig.colorbar(im0, ax=axs[0])
    # plot the output
    im1 = axs[1].imshow(output[idx1, idx2, :, :].cpu().detach().numpy())
    axs[1].set_title(f"Output: Timestep {idx1}, {idx2}. Dim {dim}")
    fig.colorbar(im1, ax=axs[1])
    # plot the target
    im2 = axs[2].imshow(target[idx1, idx2, :, :].cpu().detach().numpy())
    axs[2].set_title(f"Target: Timestep {idx1}, {idx2}. Dim {dim}")
    fig.colorbar(im2, ax=axs[2])
    # save the figure
    plt.savefig(f"{savename}_id1{idx1}_id2{idx2}.png")
    plt.close()
