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

## Differences with graphdeco-inria/gaussian-splatting

### Updates

- 2024-05-14: Improved compression scores! New results for paper v2 available on the [project website](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)
- 2024-05-02: Revised [paper v2](https://arxiv.org/pdf/2312.13299) on arXiv: Added compression of spherical harmonics, updated compression method with improved results (all attributes compressed with JPEG XL now), added qualitative comparison of additional scenes, moved compression explanation and comparison to main paper, added comparison with "Making Gaussian Splats smaller".
- 2024-02-22: The code for the sorting algorithm is now available at [fraunhoferhhi/PLAS](https://github.com/fraunhoferhhi/PLAS)
- 2024-02-21: Video comparisons for different scenes available on the [project website](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)
- 2023-12-19: Preprint available on [arXiv](https://arxiv.org/abs/2312.13299)
