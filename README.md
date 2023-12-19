## Compact 3D Scene Representation via Self-Organizing Gaussian Grids

Wieland Morgenstern, Florian Barthel, Anna Hilsmann, Peter Eisert

Abstract: *3D Gaussian Splatting has recently emerged as a highly promising technique for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it utilizes efficient rasterization allowing for very fast rendering at high-quality. However, the storage size is significantly higher, which hinders practical deployment, e.g.on resource constrained devices. In this paper, we introduce a compact scene representation organizing the parameters of 3D Gaussian Splatting (3DGS) into a 2D grid with local homogeneity,  ensuring a drastic reduction in storage requirements without compromising visual quality during rendering.
Central to our idea is the explicit exploitation of perceptual redundancies present in natural scenes. In essence,  the inherent nature of a scene allows for numerous  permutations of Gaussian parameters to equivalently represent it. To this end, we propose a novel highly parallel algorithm that regularly arranges the high-dimensional Gaussian parameters into a 2D grid while preserving their neighborhood structure. During training, we further enforce local smoothness between the sorted parameters in the grid. The uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless integration with established renderers. Our method achieves a reduction factor of 8x to 26x in size for complex scenes with no increase in training time, marking a substantial leap forward in the domain of 3D scene distribution and consumption.*

This repository is a fork of the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering".

The code for "Compact 3D Scene Representation via Self-Organizing Gaussian Grids" will become available in this repository in January 2024.
