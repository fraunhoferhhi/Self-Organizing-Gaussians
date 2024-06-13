import shutil
import sys
import os
import sys
from argparse import ArgumentParser

from compression.compression_exp import run_single_decompression

def decompress_single_to_ply(compressed_model_path):

    iter_folder_name = os.path.basename(os.path.dirname(compressed_model_path))

    decompressed_gaussians = run_single_decompression(compressed_model_path)
    decompressed_model_path = os.path.join(compressed_model_path, "decompressed_model")
    ply_path = os.path.join(decompressed_model_path, "point_cloud", iter_folder_name, "point_cloud.ply")

    os.makedirs(decompressed_model_path, exist_ok=True)

    decompressed_gaussians.save_ply(ply_path)

    # copy cfg_args file from parent/parent/parent folder to decompressed_model_path
    model_dir = os.path.dirname(os.path.dirname(os.path.dirname(compressed_model_path)))
    for file_name in ["cfg_args", "cameras.json"]:
        shutil.copyfile(os.path.join(model_dir, file_name), os.path.join(decompressed_model_path, file_name))

def decompress_all_to_ply(compressions_dir):

    for compressed_dir in os.listdir(compressions_dir):
        if not os.path.isdir(os.path.join(compressions_dir, compressed_dir)):
            continue
        decompress_single_to_ply(os.path.join(compressions_dir, compressed_dir))


def decompress():
    # example args: --compressed_model output/2023-11-14/14-01-13-blur-15/5/compression/iteration_30000/jxl_man

    # Set up command line argument parser
    parser = ArgumentParser(description="Decompression script parameters")
    parser.add_argument("--compressed_model_path", type=str)
    args_cmdline = parser.parse_args(sys.argv[1:])

    # like output/2023-11-14/14-01-13-blur-15/5/compression/iteration_30000/jxl_man
    compressed_model_path = args_cmdline.compressed_model_path

    decompress_single_to_ply(compressed_model_path)


if __name__ == "__main__":
    decompress()
