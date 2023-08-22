import numpy as np
import torch


def sample_uniform_spherical_shell(npoints: int, radii: float, shape: tuple):
    """
    Samples points uniformly from a spherical shell.

    Args:
        npoints (int): Number of points to sample.
        radii (float): Inner and outer radii of the shell.
        shape (tuple): Shape of each instance.

    Returns:
        np.array: Array of sampled points.
    """
    ndim = np.prod(shape)
    inner_radius, outer_radius = radii
    pts = []
    for i in range(npoints):
        # uniformly sample radius
        samp_radius = np.random.uniform(inner_radius, outer_radius)
        # ref: https://mathworld.wolfram.com/SpherePointPicking.html
        vec = np.random.randn(ndim)
        vec /= np.linalg.norm(vec, axis=0)
        pts.append(np.reshape(samp_radius * vec, shape))

    return np.array(pts)


def sigmoid_partition_unity(norm_of_x, shift, scale):
    """
    Partitions of unity - input is real number, output is in interval [0,1].

    Args:
        norm_of_x (float): Real number input.
        shift (float): x-coord of 0.5 point in graph of function.
        scale (float): Larger numbers make a steeper descent at shift x-coord.

    Returns:
        float: Output in interval [0,1].
    """
    return 1 / (1 + torch.exp(scale * (norm_of_x - shift)))


def linear_scale_dissipative_target(inputs, scale):
    """
    Dissipative functions - input is point x in state space (practically, subset of R^n)

    Args:
        inputs (torch.Tensor): Input point in state space.
        scale (float): Real number 0 < scale < 1 that scales down input x.

    Returns:
        torch.Tensor: Output of the dissipative function.
    """
    return scale * inputs


def part_unity_post_process(x, model, rho, diss):
    """
    Outputs prediction after post-processing according to:
        rho(|x|) * model(x) + (1 - rho(|x|)) * diss(x)

    Args:
        x (torch.Tensor): Input point as torch tensor.
        model (torch.nn.Module): Torch model.
        rho (function): Partition of unity, a map from R to [0,1].
        diss (function): Baseline dissipative map from R^n to R^n.

    Returns:
        torch.Tensor: Output of the post-processing function.
    """
    return rho(torch.norm(x)) * model(x).reshape(x.shape[0],) + (
        1 - rho(x)
    ) * diss(x)
