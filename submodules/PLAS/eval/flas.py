# Code imported from https://github.com/Visual-Computing/LAS_FLAS/blob/703ea1bdce3e0191564c70d6d4c96ccfda89a4b2/python/LAS_FLAS_DPQ_colors.ipynb
# for the purpose of comparing PLAS with FLAS.
# Copyright [2022] [Kai Uwe Barthel]
# Licensed under the Apache License, Version 2.0 (the "License");

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import time
import lap
import click

from random_grid import generate_random_colors


def plot_grid(*images, figsize=10, fignumber="Filter", titles=None, occurences=False):
    """Plots any given number of images"""
    num_plots = len(images)

    plt.close(fignumber)
    fig = plt.figure(
        figsize=(
            figsize * int(min(num_plots, 5)),
            figsize * int(max(num_plots // 5, 1)),
        ),
        num=fignumber,
    )

    for i, grid in enumerate(images):
        size = grid.shape

        if size[-1] == 1:
            if occurences:
                cmap = None
            else:
                cmap = "gray"
        else:
            cmap = None

        if len(size) == 3:
            ax = fig.add_subplot(
                ((num_plots - 1) // 5) + 1,
                min(int(num_plots % 5) + (int(num_plots // 5) * 5), 5),
                i + 1,
            )
            img = grid.reshape(*size)
            ax.imshow(np.squeeze(img), cmap=cmap, vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])

        if titles is not None:
            ax.set_title(titles[i], fontsize=figsize * 3)

    plt.show()


def squared_l2_distance(q, p):
    """Calculates the squared L2 (eucldean) distance using numpy."""
    ps = np.sum(p * p, axis=-1, keepdims=True)
    qs = np.sum(q * q, axis=-1, keepdims=True)
    distance = ps - 2 * np.matmul(p, q.T) + qs.T
    return np.maximum(distance, 0)


def low_pass_filter(image, filter_size_x, filter_size_y, wrap=False):
    """Applies a low pass filter to the current image"""
    mode = "wrap" if wrap else "reflect"

    im2 = uniform_filter1d(image, filter_size_y, axis=0, mode=mode)
    im2 = uniform_filter1d(im2, filter_size_x, axis=1, mode=mode)
    return im2


def get_positions_in_radius(pos, indices, r, nc, wrap):
    """Utility function that takes a position and returns
    a desired number of positions in the given radius"""
    if wrap:
        return get_positions_in_radius_wrapped(pos, indices, r, nc)
    else:
        return get_positions_in_radius_non_wrapped(pos, indices, r, nc)


def get_positions_in_radius_non_wrapped(pos, indices, r, nc):
    """Utility function that takes a position and returns
    a desired number of positions in the given radius"""
    H, W = indices.shape

    x = pos % W
    y = int(pos / W)

    ys = y - r
    ye = y + r + 1
    xs = x - r
    xe = x + r + 1

    # move position so the full radius is inside the images bounds
    if ys < 0:
        ys = 0
        ye = min(2 * r + 1, H)

    if ye > H:
        ye = H
        ys = max(H - 2 * r, 0)

    if xs < 0:
        xs = 0
        xe = min(2 * r + 1, W)

    if xe > W:
        xe = W
        xs = max(W - 2 * r, 0)

    # concatenate the chosen position to a 1D array
    positions = np.concatenate(indices[ys:ye, xs:xe])

    if nc is None:
        return positions

    chosen_positions = np.random.choice(
        positions, min(nc, len(positions)), replace=False
    )

    return chosen_positions


def get_positions_in_radius_wrapped(pos, extended_grid, r, nc):
    """Utility function that takes a position and returns
    a desired number of positions in the given radius"""
    H, W = extended_grid.shape

    # extended grid shape is H*2, W*2
    H, W = int(H / 2), int(W / 2)
    x = pos % W
    y = int(pos / W)

    ys = (y - r + H) % H
    ye = ys + 2 * r + 1
    xs = (x - r + W) % W
    xe = xs + 2 * r + 1

    # concatenate the chosen position to a 1D array
    positions = np.concatenate(extended_grid[ys:ye, xs:xe])

    if nc is None:
        return positions

    chosen_positions = np.random.choice(
        positions, min(nc, len(positions)), replace=False
    )

    return chosen_positions


def sort_with_flas(
    X, nc, radius_factor=0.9, wrap=False, return_time=False, randomize_X=True
):
    np.random.seed(7)  # for reproducible sortings

    # setup of required variables
    N = np.prod(X.shape[:-1])

    grid_shape = X.shape[:-1]
    H, W = grid_shape

    start_time = time.time()

    # assign input vectors to random positions on the grid
    if randomize_X:
        grid = (
            np.random.permutation(X.reshape((N, -1))).reshape((X.shape)).astype(float)
        )
    else:
        grid = X.copy()

    # reshape 2D grid to 1D
    flat_X = X.reshape((N, -1))

    # create indices array
    indices = np.arange(N).reshape(grid_shape)

    if wrap:
        # create a extended grid of size (H*2, W*2)
        indices = np.concatenate((indices, indices), axis=1)
        indices = np.concatenate((indices, indices), axis=0)

    radius_f = max(H, W) / 2 - 1  # initial radius

    while True:
        print(".", end="")

        # compute filtersize that is smaller than any side of the grid
        radius = int(radius_f)
        filter_size_x = min(W - 1, int(2 * radius + 1))
        filter_size_y = min(H - 1, int(2 * radius + 1))

        # Filter the map vectors using the actual filter radius
        grid = low_pass_filter(grid, filter_size_x, filter_size_y, wrap=wrap)
        flat_grid = grid.reshape((N, -1))

        n_iters = 2 * int(N / nc) + 1
        max_swap_radius = int(round(max(radius, (np.sqrt(nc) - 1) / 2)))

        for i in range(n_iters):
            # find random swap candicates in radius of a random position
            random_pos = np.random.choice(N, size=1)
            positions = get_positions_in_radius(
                random_pos[0], indices, max_swap_radius, nc, wrap=wrap
            )

            # calc C
            pixels = flat_X[positions]
            grid_vecs = flat_grid[positions]
            C = squared_l2_distance(pixels, grid_vecs)

            # quantization of distances speeds up assingment solver
            C = (C / C.max() * 2048).astype(int)

            # get indices of best assignments
            _, best_perm_indices, _ = lap.lapjv(C)
            # best_perm_indices, _, _= lapjv.lapjv(C)

            # assign the input vectors to their new map positions
            flat_X[positions] = pixels[best_perm_indices]

        # prepare variables for next iteration
        grid = flat_X.reshape(X.shape)

        radius_f *= radius_factor
        # break condition
        if radius_f < 1:
            break

    print("")

    duration = time.time() - start_time

    if return_time:
        return grid, duration

    print(f"Sorted with FLAS in {duration:.3f} seconds")
    return grid


def compute_spatial_distances_for_grid(grid_shape, wrap):
    """Converts a given gridshape to a grid index matrix
    and calculates the squared spatial distances"""
    if wrap:
        return compute_spatial_distances_for_grid_wrapped(grid_shape)
    else:
        return compute_spatial_distances_for_grid_non_wrapped(grid_shape)


def compute_spatial_distances_for_grid_wrapped(grid_shape):
    n_x = grid_shape[0]
    n_y = grid_shape[1]

    wrap1 = [[0, 0], [0, 0], [0, 0], [0, n_y], [0, n_y], [n_x, 0], [n_x, 0], [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0, 0], [n_x, 0], [0, 0], [0, n_y], [0, 0]]

    # create 2D position matrix with tuples, i.e. [(0,0), (0,1), ... (H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    for i in range(8):
        # look for smaller distances with wrapped coordinates
        d_i = squared_l2_distance(mat_flat + wrap1[i], mat_flat + wrap2[i])
        d = np.minimum(d, d_i)

    return d


def compute_spatial_distances_for_grid_non_wrapped(grid_shape):
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    return d


def sort_hddists_by_2d_dists(hd_dists, ld_dists):
    """sorts a matrix so that row values are sorted by the
    spatial distance and in case they are equal, by the HD distance"""
    max_hd_dist = np.max(hd_dists) * 1.0001

    ld_hd_dists = (
        hd_dists / max_hd_dist + ld_dists
    )  # add normed HD dists (0 .. 0.9999) to the 2D int dists
    ld_hd_dists = np.sort(ld_hd_dists)  # then a normal sorting of the rows can be used

    sorted_HD_D = np.fmod(ld_hd_dists, 1) * max_hd_dist

    return sorted_HD_D


def get_distance_preservation_gain(sorted_d_mat, d_mean):
    """computes the Distance Preservation Gain delta DP_k(S)"""
    # range of numbers [1, K], with K = N-1
    nums = np.arange(1, len(sorted_d_mat))

    # compute cumulative sum of neighbor distance values for all rows, shape = (N, K)
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)

    # compute average of neighbor distance values for all rows, shape = (N, K)
    d_k = cumsum / nums

    # compute average of all rows for each k, shape = (K, )
    d_k = d_k.mean(axis=0)

    # compute Distance Preservation Gain and set negative values to 0, shape = (K, )
    d_k = np.maximum((d_mean - d_k) / d_mean, 0)

    return d_k


def distance_preservation_quality(sorted_X, p=2, wrap=False):
    """computes the Distance Preservation Quality DPQ_p(S)"""
    # setup of required variables
    grid_shape = sorted_X.shape[:-1]
    N = np.prod(grid_shape)
    H, W = grid_shape
    flat_X = sorted_X.reshape((N, -1))

    # compute matrix of euclidean distances in the high dimensional space
    dists_HD = np.sqrt(squared_l2_distance(flat_X, flat_X))

    # sort HD distance matrix rows in acsending order (first value is always 0 zero now)
    sorted_D = np.sort(dists_HD, axis=1)

    # compute the expected value of the HD distance matrix
    mean_D = sorted_D[:, 1:].mean()

    # compute spatial distance matrix for each position on the 2D grid
    dists_spatial = compute_spatial_distances_for_grid(grid_shape, wrap)

    # sort rows of HD distances by the values of spatial distances
    sorted_HD_by_2D = sort_hddists_by_2d_dists(dists_HD, dists_spatial)

    # get delta DP_k values
    delta_DP_k_2D = get_distance_preservation_gain(sorted_HD_by_2D, mean_D)
    delta_DP_k_HD = get_distance_preservation_gain(sorted_D, mean_D)

    # compute p norm of DP_k values
    normed_delta_D_2D_k = np.linalg.norm(delta_DP_k_2D, ord=p)
    normed_delta_D_HD_k = np.linalg.norm(delta_DP_k_HD, ord=p)

    # DPQ(s) is the ratio between the two normed DP_k values
    DPQ = normed_delta_D_2D_k / normed_delta_D_HD_k

    return DPQ


@click.command()
@click.option("--width", default=48, help="Width of the grid")
@click.option("--height", default=48, help="Height of the grid")
def flas_grid(width, height):
    X = generate_random_colors(width, height)
    sorted_X, duration = sort_with_flas(
        X.copy(), nc=100, radius_factor=0.85, wrap=False, return_time=True
    )
    dpq = distance_preservation_quality(sorted_X, p=16)
    plot_grid(
        X,
        sorted_X,
        figsize=6,
        titles=[
            f"Random {X.shape}",
            f"{duration:0.3f}s, DPQ_16: {dpq:0.3f}",
        ],
    )


if __name__ == "__main__":
    flas_grid()
