# An example script to sort Gaussians from a .ply file with PLAS.

# The input is a .ply file with the Gaussians, and the output is a .ply file with the sorted Gaussians.
# For sorting, it is only using the 3D coordinates and the RGB colors (SH DC component) of the Gaussians.

# Note that sorting a .ply after training the model is much less efficient than sorting the Gaussians during training,
# and applying a regularization on the sorted grid. See results int Table 3 of the paper (https://arxiv.org/abs/2312.13299).

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement
import trimesh as tm
import click
import os

from plas import sort_with_plas, compute_vad


# process fewer elements for development testing
# DEBUG_TRUNCATE_ELEMENTS = 1_000_000
DEBUG_TRUNCATE_ELEMENTS = None

COORDS_SCALE = 255
RGB_SCALE = 255

C0 = 0.28209479177387814

def prune_gaussians(df, num_to_keep):
    """Very crude pruning method that uses scaling and opacity to determine the impact of a Gaussian splat.
       We need this method to drop a few Gaussians to make them fit a square image.

       For a more sophisticated method, see e.g. "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
       https://arxiv.org/abs/2311.17245
    """

    # from gaussian model:
    # self.scaling_activation = torch.exp
    # self.scaling_inverse_activation = torch.log

    # self.opacity_activation = torch.sigmoid
    # self.inverse_opacity_activation = inverse_sigmoid

    scaling_act = np.exp
    opacity_act = lambda x: 1 / (1 + np.exp(-x))

    # does this perhaps remove too many small points in the center of the scene, that form a bigger object?
    df["impact"] = scaling_act((df["scale_0"] + df["scale_1"] + df["scale_2"]).astype(np.float64)) * opacity_act(df["opacity"].astype(np.float64))

    df = df.sort_values("impact", ascending=False)
    df = df.head(num_to_keep)

    return df

def SH2RGB(sh):
    return sh * C0 + 0.5

def df_to_gs_ply(df, ply_file):
    ply_columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
       'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5',
       'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10',
       'f_rest_11', 'f_rest_12', 'f_rest_13', 'f_rest_14', 'f_rest_15',
       'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19', 'f_rest_20',
       'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24', 'f_rest_25',
       'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29', 'f_rest_30',
       'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35',
       'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39', 'f_rest_40',
       'f_rest_41', 'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity',
       'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    df_ply = df[ply_columns]

    df_ply = df_ply.astype(np.float32)
    vertex_el = PlyElement.describe(df_ply.to_records(index=False), "vertex")
    ply_data = PlyData([vertex_el], text=False)
    ply_data.write(ply_file)


def df_to_rgb_ply(df, ply_file):

    xyz = df[['x', 'y', 'z']].values

    dc_vals = df.loc[:, df.columns.str.startswith("f_dc")].values
    rgb = np.clip(SH2RGB(dc_vals), 0, 1)

    pcl = tm.PointCloud(xyz, colors=rgb)
    pcl.export(ply_file)


def gs_ply_to_df(ply_file, min_block_size):
    gaussian_ply = tm.load(ply_file)
    ply_data = gaussian_ply.metadata["_ply_raw"]["vertex"]["data"]
    df = pd.DataFrame(ply_data)
    # df.describe().drop("count")

    if DEBUG_TRUNCATE_ELEMENTS is not None:
        df = df.iloc[:DEBUG_TRUNCATE_ELEMENTS]

    num_gaussians = len(df)

    sidelen = int(np.sqrt(num_gaussians))
    sidelen = sidelen // min_block_size * min_block_size

    # throw away a few (small * transparent) Gaussians to make the number of Gaussians fit into a square image
    df = prune_gaussians(df, sidelen * sidelen)

    return df, sidelen

@click.command()
@click.option("--input-gs-ply", type=click.Path(exists=True))
@click.option("--output-gs-ply", type=click.Path())
@click.option("--output-rgb-point-cloud-ply", type=click.Path())
def sort_gaussians(input_gs_ply, output_gs_ply, output_rgb_point_cloud_ply):

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    min_block_size = 16

    df, sidelen = gs_ply_to_df(input_gs_ply, min_block_size)

    orig_vad = compute_vad(df.values.reshape(sidelen, sidelen, -1))
    print(f"VAD of ply: {orig_vad:.4f}")

    # shuffling the input to avoid getting stuck in a local minimum with the sorting
    df = df.sample(frac=1)

    shuffled_vad = compute_vad(df.values.reshape(sidelen, sidelen, -1))
    print(f"VAD of shuffled ply: {shuffled_vad:.4f}")

    # scale coords to [0, COORDS_SCALE] for sorting (original values are kept)
    coords_xyz = df[["x", "y", "z"]].values
    coords_xyz_min = coords_xyz.min()
    coords_xyz_range = coords_xyz.max() - coords_xyz_min
    coords_xyz_norm = (coords_xyz - coords_xyz_min) / coords_xyz_range
    coords_xyz_norm *= COORDS_SCALE
    coords_torch = torch.from_numpy(coords_xyz_norm).float().to(device)

    # scale colors to [0, RGB_SCALE] for sorting (original values are kept)
    dc_vals = df.loc[:, df.columns.str.startswith("f_dc")].values
    rgb = np.clip(SH2RGB(dc_vals), 0, 1) * RGB_SCALE
    rgb_torch = torch.from_numpy(rgb).float().to(device)

    # params to sort: 6D (3D coords + 3D colors)
    params = torch.cat([coords_torch, rgb_torch], dim=1)

    params_torch_grid = params.permute(1, 0).reshape(-1, sidelen, sidelen)

    sorted_coords, sorted_grid_indices = sort_with_plas(params_torch_grid, min_block_size, improvement_break=1e-4, verbose=True)

    sorted_indices = sorted_grid_indices.flatten().cpu().numpy()

    sorted_df = df.iloc[sorted_indices]

    sorted_vad = compute_vad(sorted_df.values.reshape(sidelen, sidelen, -1))
    print(f"VAD of sorted ply: {sorted_vad:.4f}")

    if output_gs_ply is not None:    
        os.makedirs(os.path.dirname(output_gs_ply), exist_ok=True)
        # full gaussians ply
        df_to_gs_ply(sorted_df, output_gs_ply)

    if output_rgb_point_cloud_ply is not None:
        os.makedirs(os.path.dirname(output_rgb_point_cloud_ply), exist_ok=True)
        # xyz + rgb colors point cloud ply
        df_to_rgb_ply(sorted_df, output_rgb_point_cloud_ply)


if __name__ == "__main__":
    sort_gaussians()

