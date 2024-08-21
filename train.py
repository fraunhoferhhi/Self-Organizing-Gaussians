#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import hydra


from omegaconf import DictConfig, OmegaConf
from utils.dict import EasyDict
from tqdm import tqdm
from copy import deepcopy
from lpipsPyTorch import lpips
from random import randint

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.wandb_utils import init_wandb, save_hist, wandb

from training_viewer import TrainingViewer

from compression.compression_exp import run_compressions, run_decompressions
from compression.decompress import decompress_all_to_ply

from splatviz_network import SplatvizNetwork


def training(cfg):

    first_iter = 0

    if not cfg.run.use_sh:
        print("use_sh not set, disabling sorting for spherical harmonics")
        cfg.sorting.weights.features_rest = 0
        cfg.neighbor_loss.weights.features_rest = 0
        for compression in cfg.compression["experiments"]:
            for i, att in enumerate(compression["attributes"]):
                if att["name"] == "_features_rest":
                    del compression["attributes"][i]

    print(f"Starting training on dataset {cfg.dataset.source_path}")

    disable_xyz_log_activation = "disable_xyz_log_activation" in cfg.optimization and cfg.optimization.disable_xyz_log_activation
    print(f"{disable_xyz_log_activation=}")

    gaussians = GaussianModel(cfg.dataset.sh_degree, disable_xyz_log_activation=disable_xyz_log_activation)
    scene = Scene(cfg.dataset, gaussians)
    gaussians.training_setup(cfg.optimization)
    if cfg.run.start_checkpoint:
        (model_params, first_iter) = torch.load(cfg.run.start_checkpoint)
        gaussians.restore(model_params, cfg.optimization)

    # ----------------------
    # SSGS
    if cfg.sorting.enabled:
        gaussians.prune_to_square_shape()
        gaussians.sort_into_grid(cfg.sorting, not cfg.run.no_progress_bar)

    debug_viewer = TrainingViewer(debug_view=cfg.local_window_debug_view.view_id)
    # ----------------------

    bg_color = [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if cfg.run.no_progress_bar:
        progress_bar = None
    else:
        progress_bar = tqdm(range(first_iter, cfg.optimization.iterations), desc="Training progress")
    first_iter += 1

    network = SplatvizNetwork()
    opt = EasyDict(cfg.optimization)
    for key, value in cfg.sorting.items():
        if key == "weights":
            for subkey, subvalue in value.items():
                opt[f"sorting_{key}_{subkey}"] = subvalue
        opt[f"sorting_{key}"] = value

    for iteration in range(first_iter, cfg.optimization.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and cfg.run.use_sh:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == cfg.debug.debug_from:
            cfg.pipeline.debug = True

        bg = torch.rand((3), device="cuda") if cfg.optimization.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, cfg.pipeline, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - cfg.optimization.lambda_dssim) * Ll1 + cfg.optimization.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ----------------------
        # SSGS
        if cfg.neighbor_loss.lambda_neighbor > 0:
            
            nb_losses = []
            wandb_log = {}
            
            attr_getter_fn = gaussians.get_activated_attr_flat if cfg.neighbor_loss.activated else gaussians.get_attr_flat

            weight_sum = sum(cfg.neighbor_loss.weights.values())
            for attr_name, attr_weight in cfg.neighbor_loss.weights.items():
                if attr_weight > 0:
                    nb_losses.append(gaussians.neighborloss_2d(attr_getter_fn(attr_name), cfg.neighbor_loss) * attr_weight / weight_sum)
                    wandb_log[f"neighbor_loss/{attr_name}"] = nb_losses[-1]
                
            nb_loss = cfg.neighbor_loss.lambda_neighbor * sum(nb_losses)
            
            if iteration % cfg.run.log_nb_loss_interval == 0:
                wandb.log(wandb_log, step=iteration)
        else:
            nb_loss = torch.tensor(0.0)
        # ----------------------

        loss += nb_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if progress_bar is not None:
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == cfg.optimization.iterations:
                    progress_bar.close()

            # Debug view
            if cfg.wandb_debug_view.interval != -1 and iteration % cfg.wandb_debug_view.interval == 0:
                if cfg.wandb_debug_view.view_enabled:
                    debug_viewer.training_view_wandb(scene, gaussians, pipe=cfg.pipeline, step=iteration, background=background)
                if cfg.wandb_debug_view.save_hist:
                    save_hist(gaussians, step=iteration)
                
            if cfg.local_window_debug_view.enabled and cfg.local_window_debug_view.interval != -1 and iteration % cfg.local_window_debug_view.interval == 0:
                debug_viewer.training_view(scene, gaussians, pipe=cfg.pipeline, background=background)
                

            # Log and save
            if iteration % cfg.run.log_training_report_interval == 0:
                wandb.log(
                    {
                        "loss/l1_loss": Ll1.item(),
                        "loss/total_loss": loss.item(),
                        "loss/nb_loss": nb_loss.item(),
                        "iter_time": iter_start.elapsed_time(iter_end),
                        "num gaussians": len(gaussians.get_xyz),
                    },
                    step=iteration
                )
                if iteration in cfg.run.test_iterations:
                    training_report(cfg, iteration, scene, gaussians, (cfg.pipeline, background), log_name="uncompressed")

            if (iteration in cfg.run.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Compression
            if (iteration in cfg.run.compress_iterations):
                print("\n[ITER {}] Compressing Gaussians".format(iteration))
                compr_path = os.path.join(cfg.dataset.model_path, "compression", f"iteration_{iteration}")
                
                # enable compression of non-sorted gaussians without affecting results
                gaussians_to_compress = deepcopy(gaussians)
                gaussians_to_compress.prune_to_square_shape()
                
                compr_results = run_compressions(gaussians_to_compress, compr_path, OmegaConf.to_container(cfg.compression))
                wandb.log(compr_results, step=iteration)

                for compr_name, decompressed_gaussians in run_decompressions(compr_path):
                    training_report(cfg, iteration, scene, decompressed_gaussians, (cfg.pipeline, background), log_name=f"cmpr_{compr_name}", log_GT=False)
                
                # decompress plys in last compression iteration
                if iteration == max(cfg.run.compress_iterations):
                    decompress_all_to_ply(compr_path)

            # Densification
            if iteration < cfg.optimization.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg.optimization.densify_from_iter and iteration % cfg.optimization.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.optimization.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=cfg.optimization.densify_grad_threshold, min_opacity=cfg.optimization.densify_min_opacity, extent=scene.cameras_extent, max_screen_size=size_threshold)

            if iteration > cfg.optimization.densify_from_iter and iteration % cfg.optimization.densification_interval == 0:
                # ----------------------
                # SSGS
                if cfg.sorting.enabled:
                    gaussians.prune_to_square_shape()
                    gaussians.sort_into_grid(cfg.sorting, not cfg.run.no_progress_bar)
                # ----------------------

            if iteration < cfg.optimization.densify_until_iter:
                if iteration % cfg.optimization.opacity_reset_interval == 0 or (
                        cfg.dataset.white_background and iteration == cfg.optimization.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < cfg.optimization.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in cfg.run.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        network.render(EasyDict(cfg.pipeline), gaussians, ema_loss_for_log, render, background, iteration, opt, cfg)


def training_report(cfg, iteration, scene, gaussians, renderArgs, log_name, log_GT=True):
    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = (
        {
            'name': 'test',
            'cameras': scene.getTestCameras()
        },
        {
            'name': 'train',
            'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        }
    )

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpipss_test = 0.0
            wandb_images = []
            wandb_gt_images = []
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(render(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if cfg.wandb_debug_view.view_enabled and idx < 5:
                    name = config['name'] + "_view_{}/render".format(viewpoint.image_name)
                    wandb_img = wandb.Image(image[None], caption=name)
                    wandb_images.append(wandb_img)
                    name = config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name)
                    wandb_img = wandb.Image(gt_image[None], caption=name)
                    wandb_gt_images.append(wandb_img)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image)
                if cfg.run.test_lpips:
                    lpipss_test += lpips(image, gt_image, net_type='vgg').item()

            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpipss_test /= len(config['cameras'])
            print(f"\n[ITER {iteration}] Evaluating {log_name} {config['name']}: L1: {l1_test:.4f} - PSNR: {psnr_test:.4f} - SSIM: {ssim_test:.4f} - LPIPS: {lpipss_test:.4f}")

            to_log = {
                    f"eval_{config['name']}_PSNR/{log_name}": psnr_test,
                    f"eval_{config['name']}_SSIM/{log_name}": ssim_test,
                    f"eval_{config['name']}_LPIPS/{log_name}": lpipss_test,
                    f"eval_{config['name']}_L1/{log_name}": l1_test,
                    f"eval_{config['name']}_renders/{log_name}": wandb_images,
            }

            if log_GT:
                to_log[f"eval_{config['name']}_gt_img/{log_name}"] = wandb_gt_images

            wandb.log(to_log, step=iteration)
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path='config', config_name='training')
def main(cfg: DictConfig):

    # Initialize system state (RNG)
    safe_state(cfg.run.quiet)
    cfg.run.wandb_url = init_wandb(cfg)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if not cfg.dataset.model_path:
        cfg.dataset.model_path = output_dir

    yaml_config = OmegaConf.to_yaml(cfg)

    training_config_yml_path = os.path.join(cfg.dataset.model_path, "training_config.yaml")
    with open(training_config_yml_path, 'w') as file:
        file.write("# Also available at .hydra/config.yaml\n")
        file.write(yaml_config)

    cfg_args_path = os.path.join(cfg.dataset.model_path, "cfg_args")
    with open(cfg_args_path, "w") as file:
        file.write(f"Namespace(model_path='{cfg.dataset.model_path}', source_path='{cfg.dataset.source_path}', images='{cfg.dataset.images}', resolution='{cfg.dataset.resolution}', sh_degree={cfg.dataset.sh_degree}, white_background={cfg.dataset.white_background}, eval={cfg.dataset.eval})")


    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.debug.detect_anomaly)
    training(cfg)

    # All done
    print("\nTraining complete.")

if __name__ == "__main__":
    main()



