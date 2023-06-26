from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import numpy as np


def create_animation(
    data: torch.tensor, time: torch.tensor, save_dir: str, mass, fps=10
):
    if isinstance(mass, str):
        mass = [mass]
    # create an animation of the density and velocity data
    tmp = data.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    im1 = ax[0].imshow(tmp[0, ..., 0], cmap="inferno")
    ax[0].set_title(f"Density Mass {mass[0]} M-sun Timestep {float(time[0]):0.3f} kyr")
    im2 = ax[1].imshow(tmp[0, ..., 1], cmap="inferno")
    ax[1].set_title("Velocity X")
    im3 = ax[2].imshow(tmp[0, ..., 2], cmap="inferno")
    ax[2].set_title("Velocity Y")
    im1.set_clim(vmin=tmp[0, ..., 0].min(), vmax=tmp[0, ..., 0].max())
    im2.set_clim(vmin=tmp[0, ..., 1].min(), vmax=tmp[0, ..., 1].max())
    im3.set_clim(vmin=tmp[0, ..., 2].min(), vmax=tmp[0, ..., 2].max())
    # add colorbar to each axis
    fig.colorbar(im1, ax=ax[0], shrink=0.5)
    fig.colorbar(im3, ax=ax[2], shrink=0.5)
    fig.colorbar(im2, ax=ax[1], shrink=0.5)

    plt.tight_layout()

    NUM_FRAMES = data.shape[0]

    def animate(i):
        im1.set_data(tmp[i, ..., 0])
        im2.set_data(tmp[i, ..., 1])
        im3.set_data(tmp[i, ..., 2])
        if i:
            im1.set_clim(vmin=tmp[i, ..., 0].min(), vmax=tmp[i, ..., 0].max())
            im2.set_clim(vmin=tmp[i, ..., 1].min(), vmax=tmp[i, ..., 1].max())
            im3.set_clim(vmin=tmp[i, ..., 2].min(), vmax=tmp[i, ..., 2].max())
        if len(mass) > 1:
            ax[0].set_title(
                f"Density Mass {mass[i]} Timestep {float(time[i]):0.3f} kyr"
            )
        else:
            ax[0].set_title(
                f"Density Mass {mass[0]} Timestep {float(time[i]):0.3f} kyr"
            )

    ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=100)
    ani.save(
        f"{save_dir}_M{mass[0]}_Movie.mp4", fps=fps, extra_args=["-vcodec", "libx264"]
    )
    plt.close()
    return None


def random_plot(
    input: torch.tensor, output: torch.tensor, target: torch.tensor, savename: str
):
    """Plots a time point of the input, output, and target"""
    # select random first and second dimension
    idx1 = np.random.randint(0, input.shape[0])
    idx2 = np.random.randint(0, input.shape[1])
    # check the shape of the input
    d = "1 D problem"
    if len(input.shape) > 4:
        d = np.random.randint(0, input.shape[-1])
        d = 0
        input = input[..., d]
        output = output[..., d]
        target = target[..., d]

    # create 1 by 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plot the input
    im0 = axs[0].imshow(input[idx1, idx2, :, :].cpu().detach().numpy())
    axs[0].set_title(f"Input: Timestep {idx1}, {idx2}. Dim {d}")
    fig.colorbar(im0, ax=axs[0])
    # plot the output
    im1 = axs[1].imshow(output[idx1, idx2, :, :].cpu().detach().numpy())
    axs[1].set_title(f"Output: Timestep {idx1}, {idx2}. Dim {d}")
    fig.colorbar(im1, ax=axs[1])
    # plot the target
    im2 = axs[2].imshow(target[idx1, idx2, :, :].cpu().detach().numpy())
    axs[2].set_title(f"Target: Timestep {idx1}, {idx2}. Dim {d}")
    fig.colorbar(im2, ax=axs[2])
    # save the figure
    plt.savefig(f"{savename}_id1{idx1}_id2{idx2}.png")
    plt.close()
