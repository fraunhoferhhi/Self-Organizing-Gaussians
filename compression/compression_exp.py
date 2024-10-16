import sys
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import get_hydra_training_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

import yaml
from dataclasses import dataclass, asdict
import pandas as pd

from compression.jpeg_xl import JpegXlCodec
from compression.npz import NpzCodec
from compression.exr import EXRCodec
from compression.png import PNGCodec

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
}

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


@dataclass
class QuantEval:
    psnr: float
    ssim: float
    lpips: float

@dataclass
class Measurement:
    name: str
    path: str
    size_bytes: int
    quant_eval: QuantEval = None

    @property
    def human_readable_byte_size(self):
        if self.size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(np.floor(np.log(self.size_bytes) / np.log(1000)))
        p = np.power(1000, i)
        s = round(self.size_bytes / p, 2)
        return f"{s}{size_name[i]}"
    
    def to_dict(self):
        d = asdict(self)
        d.pop('quant_eval')
        if self.quant_eval is not None:
            d.update(self.quant_eval.__dict__)
        d['size'] = self.human_readable_byte_size
        return d



def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    zero = coords == 0

    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    # For zero, no change is needed as transformed_coords is already initialized to zeros

    return transformed_coords

def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0

    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros

    return original_coords



def get_attr_numpy(gaussians, attr_name):
    attr_tensor = gaussians.attr_as_grid_img(attr_name)
    attr_numpy = attr_tensor.detach().cpu().numpy()
    return attr_numpy


def compress_attr(attr_config, gaussians, out_folder):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    attr_params = attr_config.get('params', {})
    
    if not attr_params:
        attr_params = {}
    
    codec = codecs[attr_method]()
    attr_np = get_attr_numpy(gaussians, attr_name)
    
    file_name = f"{attr_name}.{codec.file_ending()}"
    out_file = os.path.join(out_folder, file_name)

    if attr_config.get('contract', False):
        # sc = SceneContraction()
        # TODO take the original cuda array
        # attr = torch.tensor(attr_np, device="cuda")
        # attr_contracted = sc(attr)
        # attr_np = attr_contracted.cpu().numpy()
        attr_np = log_transform(attr_np)
    
    if "quantize" in attr_config:
        quantization = attr_config["quantize"]
        min_val = attr_np.min()
        max_val = attr_np.max()
        val_range = max_val - min_val
        # no division by zero
        if val_range == 0:
            val_range = 1
        attr_np_norm = (attr_np - min_val) / (val_range)
        qpow = 2 ** quantization
        attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
        attr_np = attr_np_quantized * (val_range) + min_val
        attr_np = attr_np.astype(np.float32)

    if attr_config.get('normalize', False):
        min_val, max_val = codec.encode_with_normalization(attr_np, attr_name, out_file, **attr_params)
        return file_name, min_val, max_val
    else:
        codec.encode(attr_np, out_file, **attr_params)
        return file_name, None, None


def decompress_attr(gaussians, attr_config, compressed_file, min_val, max_val):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    
    codec = codecs[attr_method]()

    if attr_config.get('normalize', False):
        decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
    else:
        decompressed_attr = codec.decode(compressed_file)

    if attr_config.get('contract', False):
        decompressed_attr = inverse_log_transform(decompressed_attr)

    # TODO dtype?
    # TODO to device?
    # TODO add grad?
    gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)


def run_single_compression(gaussians, experiment_out_path, experiment_config):
    compressed_min_vals = {}
    compressed_max_vals = {}

    compressed_files = {}

    total_size_bytes = 0

    for attribute in experiment_config['attributes']:
        compressed_file, min_val, max_mal = compress_attr(attribute, gaussians, experiment_out_path)
        attr_name = attribute['name']
        compressed_files[attr_name] = compressed_file
        compressed_min_vals[attr_name] = min_val
        compressed_max_vals[attr_name] = max_mal
        total_size_bytes += os.path.getsize(os.path.join(experiment_out_path, compressed_file))

    compr_info = pd.DataFrame([compressed_min_vals, compressed_max_vals, compressed_files], index=["min", "max", "file"]).T
    compr_info.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

    experiment_config['max_sh_degree'] = gaussians.max_sh_degree
    experiment_config['active_sh_degree'] = gaussians.active_sh_degree
    experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
    with open(os.path.join(experiment_out_path, "compression_config.yml"), 'w') as stream:
        yaml.dump(experiment_config, stream)

    return total_size_bytes

def run_compressions(gaussians, out_path, compr_exp_config):

    # TODO some code duplciation with run_experiments / run_roundtrip

    results = {}

    for experiment in compr_exp_config['experiments']:

        experiment_name = experiment['name']
        experiment_out_path = os.path.join(out_path, experiment_name)
        os.makedirs(experiment_out_path, exist_ok=True)

        size_bytes = run_single_compression(gaussians, experiment_out_path, experiment)
        results[f"size_bytes/cmpr_{experiment['name']}"] = size_bytes

    return results

def run_single_decompression(compressed_dir):

    compr_info = pd.read_csv(os.path.join(compressed_dir, "compression_info.csv"), index_col=0)

    with open(os.path.join(compressed_dir, "compression_config.yml"), 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    decompressed_gaussians = GaussianModel(experiment_config['max_sh_degree'], experiment_config['disable_xyz_log_activation'])
    decompressed_gaussians.active_sh_degree = experiment_config['active_sh_degree']

    for attribute in experiment_config['attributes']:
        attr_name = attribute["name"]
        # compressed_bytes = compressed_attrs[attr_name]
        compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])

        decompress_attr(decompressed_gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"])

    return decompressed_gaussians

def run_decompressions(compressions_dir):
    
    for compressed_dir in os.listdir(compressions_dir):
        compressed_dir_path = os.path.join(compressions_dir, compressed_dir)
        if not os.path.isdir(compressed_dir_path):
            continue
        yield os.path.basename(compressed_dir_path), run_single_decompression(compressed_dir_path)

def run_roundtrip(gaussians, out_path, experiment_config):

    experiment_name = experiment_config['name']
    experiment_out_path = os.path.join(out_path, experiment_name)
    os.makedirs(experiment_out_path, exist_ok=True)

    gaussians.prune_to_square_shape()
    
    total_size_bytes = run_single_compression(gaussians, experiment_out_path, experiment_config)
    
    decompressed_gaussians = run_single_decompression(experiment_out_path)

    return decompressed_gaussians, total_size_bytes, experiment_out_path






def run_experiments(training_cfg, cmdline_iteration, compr_exp_config, disable_lpips=False):

    gaussians = GaussianModel(training_cfg.dataset.sh_degree, False)

    scene = Scene(training_cfg.dataset, gaussians, load_iteration=cmdline_iteration, shuffle=False)
    iteration = scene.loaded_iter

    gaussians._xyz = gaussians.inverse_xyz_activation(gaussians._xyz.detach())

    print(f"Compressing {training_cfg.dataset.model_path} iteration {iteration}")
    out_path = os.path.join(training_cfg.dataset.model_path, "compression", f"iteration_{iteration}")
    os.makedirs(out_path, exist_ok=True)

    bg_color = [1,1,1] if training_cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    all_cameras = scene.getTestCameras() # + scene.getTrainCameras()

    def render_test_measure(gaussians_to_render):

        with torch.inference_mode():
            psnrs = []
            ssims = []
            lpipss = []

            for idx, view in enumerate(all_cameras):
                rendering = render(view, gaussians_to_render, training_cfg.pipeline, background)["render"]
                gt = view.original_image[0:3, :, :]
                psnrs.append(psnr(rendering, gt).cpu().numpy())
                ssims.append(ssim(rendering, gt).cpu().numpy())
                if disable_lpips:
                    lpipss.append(np.nan)
                else:
                    lpipss.append(lpips(rendering, gt, net_type='vgg').cpu().numpy())
        
        return QuantEval(psnr=np.mean(psnrs), ssim=np.mean(ssims), lpips=np.mean(lpipss))

    exp_results = []

    original_eval = render_test_measure(gaussians)
    exp_results.append(Measurement(name="PLY", path=scene.loaded_gaussian_ply, size_bytes=os.path.getsize(scene.loaded_gaussian_ply), quant_eval=original_eval))

    for experiment in compr_exp_config['experiments']:
        gaussians_roundtrip, compressed_size_bytes, exp_out_path = run_roundtrip(gaussians, out_path, experiment)
        rendered_eval = render_test_measure(gaussians_roundtrip)
        meas = Measurement(name=experiment['name'], path=exp_out_path, size_bytes=compressed_size_bytes, quant_eval=rendered_eval)
        print(meas)
        exp_results.append(meas)

    exp_df = pd.DataFrame([m.to_dict() for m in exp_results])

    sorted_columns_for_easy_comparison = ['name', 'size', 'psnr', 'ssim', 'lpips', 'path', 'size_bytes']

    assert len(exp_df.columns) == len(sorted_columns_for_easy_comparison), "Hey, you added a column to the dataframe, please add it to the sorted_columns_for_easy_comparison list as well"

    exp_df = exp_df[sorted_columns_for_easy_comparison]
    exp_df.to_csv(os.path.join(out_path, "results.csv"), index=False)
    return exp_df
        



def load_config(config_path: str):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def compression_exp():
    # example args: --model_path=../models/truck --iteration 10000 --compression_config compression/configs/jpeg_xl.yml [--results_csv results.csv] [--disable_lpips]

    parser = ArgumentParser(description="Compression script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--compression_config", type=str)
    parser.add_argument("--results_csv", type=str)
    parser.add_argument("--results_tex", type=str)
    parser.add_argument("--disable_lpips", action="store_true")
    
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    iteration = args_cmdline.iteration
    model_path = args_cmdline.model_path

    compr_exp_config = load_config(args_cmdline.compression_config)

    training_cfg = get_hydra_training_args(model_path)

    training_cfg.dataset.model_path = model_path
    training_cfg.dataset.source_path = args_cmdline.source_path

    disable_lpips = args_cmdline.disable_lpips

    results_csv = args_cmdline.results_csv
    results_tex = args_cmdline.results_tex

    exp_df = run_experiments(training_cfg, iteration, compr_exp_config, disable_lpips=disable_lpips)
    print(exp_df)

    if results_csv:
        csv_dirname = os.path.dirname(results_csv)
        if csv_dirname:
            os.makedirs(csv_dirname, exist_ok=True)
        exp_df.to_csv(results_csv, index=False)

    if results_tex:
        tex_dirname = os.path.dirname(results_tex)
        if tex_dirname:
            os.makedirs(tex_dirname, exist_ok=True)
        exp_df.to_latex(results_tex, index=False,
                        columns=["name", "psnr", "ssim", "lpips", "size_bytes"],
                        header=["Name", "PSNR $\\uparrow$", "SSIM $\\uparrow$", "LPIPS $\\downarrow$", "Size (MB)"],
                        formatters={"size_bytes": lambda x: f"{x / 1000 / 1000:.2f}", "psnr": lambda x: f"{x:.2f}", "ssim": lambda x: f"{x:.3f}", "lpips": lambda x: f"{x:.3f}"}
                        )

    
    

if __name__ == "__main__":
    compression_exp()
