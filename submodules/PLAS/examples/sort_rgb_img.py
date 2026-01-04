# Load an image and sort it using the PLAS algorithm
# The output is saved in the same directory as the input image, with the suffix "_sorted.png" added to the filename.

from PIL import Image
from plas import sort_with_plas
import math
import torch
import torchvision
import click
import os

from plas import compute_vad

@click.command()
@click.option("--img-path", type=click.Path(exists=True))
@click.option("--shuffle/--no-shuffle", default=True)
def sort_image(img_path, shuffle):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Read an image and put it on the GPU
    image = Image.open(img_path).convert("RGB")
    tensor_image = torchvision.transforms.ToTensor()(image).to(device)

    # Currently only support square images, so
    sidelen = int(math.sqrt(tensor_image.shape[1] * tensor_image.shape[2]))

    # truncate the image
    img_trunc = tensor_image.flatten(start_dim=1)[:, :sidelen * sidelen]

    # TODO: not really a useful vad with the truncated reshaped image
    # -> fix after supporting non-square images
    vad_img_trunc = compute_vad(img_trunc.reshape(-1, sidelen, sidelen).permute(1, 2, 0).cpu().numpy() * 255)

    if shuffle:
        # shuffle the image to avoid local minimum
        img_trunc = img_trunc[:, torch.randperm(sidelen * sidelen)]

        # TODO: see above for non-square images
        vad_img_trunc_shuf = compute_vad(img_trunc.reshape(-1, sidelen, sidelen).permute(1, 2, 0).cpu().numpy() * 255)

    # reshape the image to be a square
    img_trunc_shuf_sq = img_trunc.reshape(-1, sidelen, sidelen)

    sorted_img, grid_indices = sort_with_plas(img_trunc_shuf_sq, improvement_break=1e-4, border_type_x="reflect", border_type_y="reflect", verbose=True)

    output_file = os.path.basename(img_path).split(".")[0]
    if shuffle:
        output_file += "_shuffled"
    output_file += "_sorted.png"

    torchvision.utils.save_image(sorted_img, os.path.join(os.path.dirname(img_path), output_file))

    vad_sorted_img = compute_vad(sorted_img.permute(1, 2, 0).cpu().numpy() * 255)

    print(f"VAD of image: {vad_img_trunc:.4f}")
    if shuffle:
        print(f"VAD of shuffled image: {vad_img_trunc_shuf:.4f}")
    print(f"VAD of sorted image: {vad_sorted_img:.4f}")

if __name__ == "__main__":
    sort_image()