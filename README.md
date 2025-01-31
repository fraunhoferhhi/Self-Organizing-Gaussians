<div align="center">
   <h3 align="center">Compact 3D Scene Representation via Self-Organizing Gaussian Grids</h3>
   <br />

  <p align="center">
   <img src="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/static/images/teaser.png" alt="Teaser of the publication. Millions of Gaussians at 174 MB with a PSNR of 24.90 are sorted into 2D attribute grids, stored at 17 MB with the same PSNR">
    <br />
    <br />
    <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/"><strong>Project Page</strong></a>
    ·
    <a href="https://arxiv.org/abs/2312.13299" target="_blank"><strong>arXiv</strong></a>
  </p>

</div>

### Code

This repository is a fork of the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering".

The code for "Compact 3D Scene Representation via Self-Organizing Gaussian Grids" consists of multiple parts. The multi-dimensional sorting algorithm, PLAS, is available under the Apache License at [fraunhoferhhi/PLAS](https://github.com/fraunhoferhhi/PLAS).

The integration of the sorting, the smoothness regularization and the compression code for training and compressing 3D scenes is available in this repository.

## Cloning the Repository

The repository contains submodules, thus please check it out with

```shell
# SSH
git clone git@github.com:fraunhoferhhi/Self-Organizing-Gaussians.git --recursive
```

or

```shell
# HTTPS
git clone https://github.com/fraunhoferhhi/Self-Organizing-Gaussians.git --recursive
```

## Python Environment

The code is using a few additional Python packages on top of graphdeco-inria/gaussian-splatting. We provide an extended environment.yml:

Installation with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```shell
micromamba env create --file environment.yml --channel-priority flexible -y
micromamba activate sogs
```

## Example training

Download a dataset, e.g. [T&T](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

The train.py script expects a name to a .yaml config file in the [config/](config/) folder. All parameters for the run are by default loaded from the yaml file. An example launch file can be found in .vscode/launch.json, for launching from Visual Studio Code.

Example:

```shell
python train.py \
  --config-name ours_q_sh_local_test \
  hydra.run.dir=/data/output/${now:%Y-%m-%d}/${now:%H-%M-%S}-${run.name} \
  dataset.source_path=/data/gaussian_splatting/tandt_db/tandt/truck \
  run.no_progress_bar=false \
  run.name=vs-code-debug
```

The parameter configurations can be overriden in the launch as shown (using [Hydra](https://hydra.cc/)).

## Pre-Trained Models & Evaluation

Trained and compressed scenes are available for download in the [ECCV 2024 release](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/releases/tag/eccv-2024-data).

The script at [eval/download_eval.sh](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/blob/main/eval/download_eval.sh) will automatically:
* download the pre-trained scenes with and without spherical harmonics
* measure size on disk and number of Gaussians of the compressed scenes
* decompress the scenes into .ply
* render the test images for each scene, using the original 3DGS code
* compute the metrics (PSNR, SSIM, LPIPS) for all test images
* gather the results in .csv, in the format of the [3DGS compression survey](https://w-m.github.io/3dgs-compression-survey/)

The evaluation results can be found in [results/](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/blob/main/results/).

## Differences with graphdeco-inria/gaussian-splatting

Code differences can be found in this diff: https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/pull/1/files

### Usage

- different command-line interface for train.py (using Hydra)
- wandb.ai used for logging

### Code extensions

- post-training quantization, compression/decompression
- xyz log activation (gaussian_model.py)
- grid sorting, neighbor loss (gaussian_model.py)
- option to disable spherical harmonics

## Citation

If you use our method in your research, please cite our paper. The paper was presented at ECCV 2024 and [published](https://doi.org/10.1007/978-3-031-73013-9_2) in the official proceedings in 2025. You can use the following BibTeX entry:

```bibtex
@InProceedings{morgenstern2024compact,
  author    = {Wieland Morgenstern and Florian Barthel and Anna Hilsmann and Peter Eisert},
  title     = {Compact 3D Scene Representation via Self-Organizing Gaussian Grids},
  booktitle = {Computer Vision -- {ECCV} 2024},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {18--34},
  doi       = {10.1007/978-3-031-73013-9_2},
  url       = {https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/},
}
```

## Updates

- 2024-12-03: Freezing package versions in environment.yml, particularly imagecodecs. There is a regression in file size with later imagecodecs versions, see [#3](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/issues/3).
- 2024-11-28: Add [ECCV 2024 Redux Talk](https://www.youtube.com/watch?v=nb5U9xfx7-w), [View Dependent Podcast](https://www.youtube.com/watch?v=Y0O6R0Keywg) and proceedings .bib to project page.
- 2024-10-30: Update project page with reduction factors from updated metric computation -> 19.9x to 39.5x compression over 3DGS.
- 2024-09-16: Script to compute per-scene metrics from uploaded models (see *Pre-Trained Models & Evaluation*). This fixes issues in the metric computation, previously done from Weights & Biases runs: *Dr Johnson* now correctly attributed to DeepBlending dataset (was: *T&T*); Quality loss from quantization and compression losses correctly incorporated.
- 2024-08-22: Released pre-trained, [compressed scenes](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/releases/tag/eccv-2024-data)
- 2024-07-09: Project website updated with TLDR, contributions, insights and comparison to concurrent methods
- 2024-07-01: Our work was accepted at **ECCV 2024** 🥳
- 2024-06-13: Training code available
- 2024-05-14: Improved compression scores! New results for paper v2 available on the [project website](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)
- 2024-05-02: Revised [paper v2](https://arxiv.org/pdf/2312.13299) on arXiv: Added compression of spherical harmonics, updated compression method with improved results (all attributes compressed with JPEG XL now), added qualitative comparison of additional scenes, moved compression explanation and comparison to main paper, added comparison with "Making Gaussian Splats smaller".
- 2024-02-22: The code for the sorting algorithm is now available at [fraunhoferhhi/PLAS](https://github.com/fraunhoferhhi/PLAS)
- 2024-02-21: Video comparisons for different scenes available on the [project website](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)
- 2023-12-19: Preprint available on [arXiv](https://arxiv.org/abs/2312.13299)

