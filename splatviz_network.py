import copy
from typing import Any

import torch
import numpy as np
import traceback
import socket
import json
from scene.cameras import MiniCam
from scene import GaussianModel


class SplatvizNetwork:
    def __init__(self, host="127.0.0.1", port=6009):
        self.slider = None
        self.edit_text = None
        self.custom_cam = None
        self.scaling_modifier = None
        self.keep_alive = None
        self.do_rot_scale_python = None
        self.do_shs_python = None
        self.do_training = None
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None
        self.addr = None
        print(f"Creating splatviz network connector for host={host} and port={port}")
        self.stop_at_value = -1

    def try_connect(self):
        try:
            self.conn, self.addr = self.listener.accept()
            print(f"\nConnected to splatviz at {self.addr}")
            self.conn.settimeout(None)
        except Exception as inst:
            pass

    def read(self):
        messageLength = self.conn.recv(4)
        expected_bytes = int.from_bytes(messageLength, "little")

        current_bytes = 0
        try_counter = 10
        counter = 0
        message = bytes()
        while current_bytes < expected_bytes:
            message += self.conn.recv(expected_bytes - current_bytes)
            current_bytes = len(message)
            counter += 1
            if counter > try_counter:
                print("Package loss")
                break
        return json.loads(message.decode("utf-8"))

    def send(self, rendered_image, training_stats, grid_image):
        """Send a training snapshot to the client.

        Args:
            net_image (torch.Tensor): Rendering of the requested view on the current scene.
            training_stats (dict): Some statistics about the current training state.
            grid_image (torch.Tensor): Visualization of the requested attributes in the current sorted grid. The values of the tensor have to be between 0 and 1.
        """
        net_image_bytes = memoryview(
            (torch.clamp(rendered_image, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        self.conn.sendall(net_image_bytes)

        training_stats = json.dumps(training_stats)
        self.conn.sendall(len(training_stats).to_bytes(4, "little"))
        self.conn.sendall(training_stats.encode())

        if grid_image is None:
            no_grid_bytes = 0
            self.conn.sendall(no_grid_bytes.to_bytes(4, "little"))
        else:
            net_grid_bytes = memoryview(
                (grid_image * 255).byte().contiguous().cpu().numpy()
            )
            self.conn.sendall(net_grid_bytes.nbytes.to_bytes(4, "little"))
            grid_side_len, channels = grid_image.shape[1:]
            self.conn.sendall(grid_side_len.to_bytes(4, "little"))
            self.conn.sendall(channels.to_bytes(4, "little"))
            self.conn.sendall(net_grid_bytes)
            

    def receive(self):
        message = self.read()
        width = message["resolution_x"]
        height = message["resolution_y"]
        if width != 0 and height != 0:
            try:
                self.do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                self.do_shs_python = bool(message["shs_python"])
                self.do_rot_scale_python = bool(message["rot_scale_python"])
                self.keep_alive = bool(message["keep_alive"])
                self.scaling_modifer = message["scaling_modifier"]
                world_view_transform = torch.reshape(
                    torch.tensor(message["view_matrix"]), (4, 4)
                ).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]
                full_proj_transform = torch.reshape(
                    torch.tensor(message["view_projection_matrix"]), (4, 4)
                ).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                self.custom_cam = MiniCam(
                    width,
                    height,
                    fovy,
                    fovx,
                    znear,
                    zfar,
                    world_view_transform,
                    full_proj_transform,
                )
                self.edit_text = message["edit_text"]
                self.slider = message["slider"]
                self.stop_at_value = message["stop_at_value"]
                self.single_training_step = message["single_training_step"]
                self.grid_attr = message["grid_attr"]
            except Exception as e:
                traceback.print_exc()
                raise e

    def render(
        self, pipe, gaussians, loss, render, background, iteration, opt, config=None
    ):
        if self.conn == None:
            self.try_connect()
        while self.conn != None:
            edit_error = ""
            try:
                net_image_bytes = None
                net_grid_bytes = None
                self.receive()
                pipe.convert_SHs_python = self.do_shs_python
                pipe.compute_cov3D_python = self.do_rot_scale_python
                if len(self.edit_text) > 0:
                    gs = copy.deepcopy(gaussians)
                    slider = EasyDict(self.slider)
                    try:
                        exec(self.edit_text)
                    except Exception as e:
                        edit_error = str(e)
                else:
                    gs = gaussians

                if self.custom_cam != None:
                    with torch.no_grad():
                        net_image = render(
                            self.custom_cam, gs, pipe, background, self.scaling_modifer
                        )["render"]

                training_stats = {
                    "loss": loss,
                    "iteration": iteration,
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "sh_degree": gaussians.active_sh_degree,
                    "train_params": vars(opt),
                    "error": edit_error,
                    "paused": self.stop_at_value == iteration,
                }

                grid_image = None
                if (
                    "sorting_enabled" in opt
                    and opt.sorting_enabled
                    and self.grid_attr is not None
                ):
                    grid_image = getattr(self, f"get{self.grid_attr}_grid")(
                        gaussians, config.neighbor_loss.activated
                    )

                self.send(net_image, training_stats, grid_image)
                if (
                    self.do_training
                    and ((iteration < int(opt.iterations)) or not self.keep_alive)
                    and self.stop_at_value != iteration
                ):
                    break
                if self.single_training_step:
                    break

            except Exception as e:
                print(e)
                self.conn = None

    def get_xyz_grid(self, gaussians: GaussianModel, activated: bool):
        grid_image = gaussians.get_xyz if activated else gaussians._xyz
        grid_image = gaussians.get_xyz
        grid_image = self.clamp_to_two_std_and_squash_to_0_1(grid_image)
        grid_side_len = int(np.sqrt(grid_image.shape[0]))
        return grid_image.reshape(grid_side_len, grid_side_len, 3)

    def get_features_dc_grid(self, gaussians: GaussianModel, activated: bool):
        pass

    def get_features_rest_grid(self, gaussians: GaussianModel, activated: bool):
        pass

    def get_scaling_grid(self, gaussians: GaussianModel, activated: bool):
        pass

    def get_rotation_grid(self, gaussians: GaussianModel, activated: bool):
        pass

    def get_opacity_grid(self, gaussians: GaussianModel, activated: bool):
        grid_side_len = int(np.sqrt(gaussians._opacity.shape[0]))
        if activated:
            grid_image = gaussians.get_opacity.detach()
        else:
            grid_image = self.clamp_to_two_std_and_squash_to_0_1(gaussians._opacity.clone().detach())
            if grid_image.requires_grad:
                print("Opacity grid requires grad")
        return grid_image.reshape(grid_side_len, grid_side_len, 1)

    def clamp_to_two_std_and_squash_to_0_1(self, grid_image):
        mean, std = grid_image.mean(), grid_image.std()
        grid_image -= mean
        grid_image.clamp(min=-2 * std, max=2 * std)
        grid_image += 2 * std
        grid_image /= 4 * std
        return grid_image


class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
