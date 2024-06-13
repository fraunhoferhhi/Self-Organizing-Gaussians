import wandb
import os
from omegaconf import DictConfig, OmegaConf

from scene import GaussianModel


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def init_wandb(cfg):

    if os.path.exists("/mnt/output"):
        wandb_dir = "/mnt/output/wandb_out"
        os.makedirs(wandb_dir, exist_ok=True)
    else:
        wandb_dir = "wandb_out"

    config_dict = OmegaConf.to_container(cfg, resolve=True)


    run = wandb.init(
        project="ssgs",
        config=config_dict,
        dir=wandb_dir,
        group=cfg.run.group,
        name=cfg.run.name,
        tags=cfg.run.tags,
    )

    return run.url


def save_hist(gaussian: GaussianModel, step, num_bins=200):
    hist_dict = {}
    for attribute in ["_features_rest", "_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"]:
        att = getattr(gaussian, attribute).flatten()
        hist = wandb.Histogram(att.cpu().numpy(), num_bins=num_bins)
        hist_dict["hist/" + attribute[1:]] = hist
    wandb.log(hist_dict, step=step)
