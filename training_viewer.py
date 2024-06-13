import cv2
import numpy as np

from utils.quaternion import quaternion_to_matrix, matrix_to_rotation_6d
from utils.sh_utils import SH2RGB
from screeninfo import get_monitors
import torch
import wandb

from scene.gaussian_model import GaussianModel
from dataclasses import dataclass
from gaussian_renderer import render


def dcn(x: torch.tensor, normalize=False):
    if normalize:
        x = (x - x.min()) / (x.min() - x.max())
    return x.detach().cpu().numpy()


def organize_windows(window_names):
    # Get screen width and height
    monitor = get_monitors()[0]  # Assuming you have only one monitor
    screen_width, screen_height = monitor.width, monitor.height

    # Grid dimensions (3x3)
    grid_cols = 3

    # Calculate window width and height
    window_width = screen_width // 2 // grid_cols
    window_height = window_width

    min_y = 64

    # Loop through your windows and position them
    for i, window_name in enumerate(window_names):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Calculate position
        col = i % grid_cols
        row = i // grid_cols
        x = screen_width // 2 + col * window_width
        y = min_y + row * (window_height + min_y)

        # Set window position and size
        cv2.moveWindow(window_name, x, y)
        cv2.resizeWindow(window_name, window_width, window_height)


def show_grad_img(gaussians, grad, name):
    grad = torch.norm(grad, dim=-1)

    gradn = grad / grad.max()
    gradn = gradn ** 0.3
    gradn = gaussians.as_grid_img(gradn)
    # grad = cv2.cvtColor(grad, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, gradn.cpu().numpy())


@dataclass
class TrainingViewer:
    has_updated: bool = False
    debug_view: int = 0

    def training_view(self, scene, gaussians, pipe, background=None):

        if not self.has_updated:
            # organize_windows(["xyz", "rgb", "grads_xyz_accum", "opacity", "rotation", "scale", "grad_xyz", "grad_rgb"])
            organize_windows(["xyz", "rgb", "grads_xyz_accum", "opacity", "rotation 3:6", "rotation 0:3", "scale"])

        viewpoint_cam = scene.getTrainCameras().copy()[self.debug_view]
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        img = image.moveaxis(0, -1).detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("debug view", img)

        xyzs = gaussians.as_grid_img(gaussians._xyz)
        rgbs = gaussians.as_grid_img(SH2RGB(gaussians._features_dc))

        xyzs_norm = (xyzs - xyzs.min()) / (xyzs.max() - xyzs.min())
        cv2.imshow("xyz", xyzs_norm.detach().cpu().numpy())

        rgbs_norm = torch.clamp(rgbs, 0.0, 1.0)
        cv2.imshow("rgb", rgbs_norm.detach().cpu().numpy())

        MIN_DISPLAY_SCALE = -10
        MAX_DISPLAY_SCALE = 3
        scales = gaussians.as_grid_img(gaussians._scaling)
        display_scales = torch.clamp(scales, MIN_DISPLAY_SCALE, MAX_DISPLAY_SCALE)
        scales_norm = (display_scales - MIN_DISPLAY_SCALE) / (MAX_DISPLAY_SCALE - MIN_DISPLAY_SCALE)
        cv2.imshow("scale", scales_norm.detach().cpu().numpy())

        MIN_DISPLAY_OPACITY = -5
        MAX_DISPLAY_OPACITY = 8
        opacities = gaussians.as_grid_img(gaussians._opacity)
        display_opacities = torch.clamp(opacities, MIN_DISPLAY_OPACITY, MAX_DISPLAY_OPACITY)
        opacities_norm = (display_opacities - MIN_DISPLAY_OPACITY) / (MAX_DISPLAY_OPACITY - MIN_DISPLAY_OPACITY)
        cv2.imshow("opacity", opacities_norm.detach().cpu().numpy())

        # quaternions = gaussians._rotation
        # euler_angles = tgm.quaternion_to_angle_axis(quaternions)
        # euler_norm = (euler_angles + np.pi) / (2 * np.pi)
        # euler_img = gaussians.as_grid_img(euler_norm)

        quaternions = gaussians._rotation
        matrix = quaternion_to_matrix(quaternions)
        euler_angles = matrix_to_rotation_6d(matrix)  # , convention="XYZ")
        euler_norm = (euler_angles + torch.pi) / (2 * torch.pi)
        euler_img_03 = gaussians.as_grid_img(euler_norm[..., :3])
        euler_img_36 = gaussians.as_grid_img(euler_norm[..., 3:])
        cv2.imshow("rotation 0:3", euler_img_03.detach().cpu().numpy())
        cv2.imshow("rotation 3:6", euler_img_36.detach().cpu().numpy())

        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        grads_norm = (grads - grads.min()) / (grads.max() - grads.min())
        grads_img = gaussians.as_grid_img(grads_norm)
        cv2.imshow("grads_xyz_accum", grads_img.detach().cpu().numpy())

        if not self.has_updated:
            # while cv2.waitKey(1) != 32:
            #     pass
            self.has_updated = True

        cv2.waitKey(1)

    def training_view_wandb(self, scene, gaussians: GaussianModel, step, pipe, background=None):

        # images are now rendered in evaluation
        # viewpoint_cam = scene.getTrainCameras().copy()[self.debug_view]
        # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # image = render_pkg["render"]
        # img = dcn(image.moveaxis(0, -1))
        # img = np.clip(img, 0, 1)
        # img = wandb.Image(img, caption="debug view")

        xyzs = gaussians.as_grid_img(gaussians._xyz)
        xyzs_norm = (xyzs - xyzs.min()) / (xyzs.max() - xyzs.min())
        xyz_img = wandb.Image(dcn(xyzs_norm), caption="XYZ")

        rgbs = gaussians.as_grid_img(SH2RGB(gaussians._features_dc))
        rgbs_norm = torch.clamp(rgbs, 0.0, 1.0)
        rgb_img = wandb.Image(dcn(rgbs_norm), caption="RGB")

        MIN_DISPLAY_SCALE = -10
        MAX_DISPLAY_SCALE = 3
        scales = gaussians.as_grid_img(gaussians._scaling)
        display_scales = torch.clamp(scales, MIN_DISPLAY_SCALE, MAX_DISPLAY_SCALE)
        scales_norm = (display_scales - MIN_DISPLAY_SCALE) / (MAX_DISPLAY_SCALE - MIN_DISPLAY_SCALE)
        scale_img = wandb.Image(dcn(scales_norm), caption="scale")

        MIN_DISPLAY_OPACITY = -5
        MAX_DISPLAY_OPACITY = 8
        opacities = gaussians.as_grid_img(gaussians._opacity)
        display_opacities = torch.clamp(opacities, MIN_DISPLAY_OPACITY, MAX_DISPLAY_OPACITY)
        opacities_norm = (display_opacities - MIN_DISPLAY_OPACITY) / (MAX_DISPLAY_OPACITY - MIN_DISPLAY_OPACITY)
        opacity_img = wandb.Image(dcn(opacities_norm), caption="opacity")

        # quaternions = gaussians._rotation
        # euler_angles = tgm.quaternion_to_angle_axis(quaternions)
        # euler_norm = (euler_angles + np.pi) / (2 * np.pi)
        # euler_img = gaussians.as_grid_img(euler_norm)

        quaternions = gaussians._rotation
        matrix = quaternion_to_matrix(quaternions)
        euler_angles = matrix_to_rotation_6d(matrix)
        euler_norm = (euler_angles + torch.pi) / (2 * torch.pi)
        euler_img_03 = gaussians.as_grid_img(euler_norm[..., :3])
        euler_img_36 = gaussians.as_grid_img(euler_norm[..., 3:])
        rotation_03_img = wandb.Image(dcn(euler_img_03), caption="rotation 0:3")
        rotation_36_img = wandb.Image(dcn(euler_img_36), caption="rotation 3:6")

        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        grads_norm = (grads - grads.min()) / (grads.max() - grads.min())
        grads_img = gaussians.as_grid_img(grads_norm)
        grads_xyz_accum_img = wandb.Image(dcn(grads_img), caption="grads_xyz_accum")

        to_log = {
                "grid": [
                    xyz_img,
                    rgb_img,
                    scale_img,
                    opacity_img,
                    rotation_03_img,
                    rotation_36_img,
                    grads_xyz_accum_img
                ]   
            }

        if gaussians.max_sh_degree > 0:
            sh_composed = self.sh_pyramid(gaussians)
            sh_composed_img = wandb.Image(sh_composed)
            to_log["spherical harmonics"] =  sh_composed_img

        wandb.log(to_log, step=step)

    def sh_pyramid(self, gaussians):
        w = gaussians.grid_sidelen
        sh = dcn(gaussians.get_features, normalize=True)
        sh = np.reshape(sh, [w, w, sh.shape[1], sh.shape[2]])
        sh_composed = np.zeros([w * 4, w * 7, 3])
        sh_composed[0 * w:1 * w, 3 * w:4 * w] = sh[:, :, 0, :]

        if gaussians.active_sh_degree > 1:
            sh_composed[1 * w:2 * w, 2 * w:5 * w] = np.concatenate(sh[:, :, 1:4, :].transpose(2, 0, 1, 3),
                                                                axis=0).transpose(1, 0, 2)
            sh_composed[2 * w:3 * w, 1 * w:6 * w] = np.concatenate(sh[:, :, 4:9, :].transpose(2, 0, 1, 3),
                                                                axis=0).transpose(1, 0, 2)
            sh_composed[3 * w:4 * w, 0 * w:7 * w] = np.concatenate(sh[:, :, 9:, :].transpose(2, 0, 1, 3), axis=0).transpose(
                1, 0, 2)
        return sh_composed
