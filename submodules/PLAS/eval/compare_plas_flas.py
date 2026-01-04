# Compare PLAS and FLAS on a random image
# Useful to develop and debug parallel sorting (against FLAS, which has very high quality)

# Shows the random image, PLAS sorted and FLAS sorted images
# Can write the results as PNGs to a directory
# Will also print the DPQ (for img <= 64x64) and VAD of the original, PLAS and FLAS sorted images to the console

import cv2
import numpy as np
import random
import torch
import click
import os

from flas import (
    generate_random_colors,
    sort_with_flas,
    distance_preservation_quality,
)

from plas import sort_with_plas, compute_vad


def imshow_cv2(name, img):
    if img.shape[0] < 512:
        img = cv2.resize(
            img,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow(name, img)


def cv2_to_torch_f(img, device):
    # we are sorting floats on the torch device,
    # also make the color channel the first dimension (C, H, W)
    return torch.from_numpy(img).permute(2, 0, 1).float().to(device)


def torch_to_cv2_u8(img):
    return img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def imshow_torch(name, img):
    imshow_cv2(name, torch_to_cv2_u8(img))


@click.command()
@click.option("--size", default=256, help="Size of the image")
@click.option("--show-vis/--no-show-vis", default=True, help="Show visualizations")
@click.option("--compute-plas/--no-compute-plas", default=True, help="Compute PLAS")
@click.option("--compute-flas/--no-compute-flas", default=True, help="Compute FLAS")
@click.option("--output-dir", help="Output directory")
def pssm(size, show_vis, compute_plas, compute_flas, output_dir):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print(f"Using {device=}")

    print(f"Size: {size} x {size}")

    params_bgr = generate_random_colors(size, size)

    # (channels, height, width)
    params = cv2_to_torch_f(params_bgr, device)
    assert params.shape[1] == params.shape[2]

    org_params = params.clone()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"{size}_orig.png"),
            params_bgr,
        )

    if show_vis:
        imshow_torch("Original", params)
        cv2.waitKey(20)

    # --------------------------------------------
    # PLAS
    if compute_plas:
        plas_sorted_params, plas_sorted_indices = sort_with_plas(
            org_params.clone(), border_type_x="reflect", improvement_break=1e-4, seed=42, verbose=True
        )
        if show_vis:
            imshow_torch(
                "PLAS",
                plas_sorted_params,
            )
            cv2.waitKey(20)

        assert (
            org_params.to(torch.int64).sum().item()
            == plas_sorted_params.to(torch.int64).sum().item()
        )

        if output_dir is not None:
            cv2.imwrite(
                os.path.join(output_dir, f"{size}_sorted_PLAS.png"),
                torch_to_cv2_u8(plas_sorted_params),
            )

    # --------------------------------------------
    # FLAS
    if compute_flas:
        flas_sorted, flas_time = sort_with_flas(
            org_params.permute(1, 2, 0).cpu().numpy(),
            nc=100,
            radius_factor=0.9,
            return_time=True,
            randomize_X=False,
        )
        if show_vis:
            imshow_cv2(
                "FLAS",
                flas_sorted.astype(np.uint8),
            )
            cv2.waitKey(20)

        if output_dir is not None:
            cv2.imwrite(
                os.path.join(output_dir, f"{size}_sorted_FLAS.png"),
                flas_sorted.astype(np.uint8),
            )

        print(f"FLAS took {flas_time:.2f}s")

    # --------------------------------------------
    # DPQ
    # DPQ is taking too long for larger images
    if size <= 64:
        if compute_plas:
            plas_dpq = distance_preservation_quality(
                torch_to_cv2_u8(plas_sorted_params), p=16
            )
            print(f"PLAS DPQ: {plas_dpq:.4f}", end=", ")

        if compute_flas:
            flas_dpq = distance_preservation_quality(flas_sorted, p=16)
            print(f"FLAS DPQ: {flas_dpq:.4f}", end=", ")

        org_dpq = distance_preservation_quality(params_bgr, p=16)
        print(f"Org DPQ: {org_dpq:.4f}")

    if compute_plas:
        plas_vad = compute_vad(torch_to_cv2_u8(plas_sorted_params))
        print(f"PLAS vad: {plas_vad:.4f}", end=", ")

    if compute_flas:
        flas_vad = compute_vad(flas_sorted)
        print(f"FLAS vad: {flas_vad:.4f}", end=", ")

    org_var = compute_vad(params_bgr)
    print(f"Org vad: {org_var:.4f}")

    if show_vis:
        cv2.waitKey(0)


if __name__ == "__main__":
    pssm()
