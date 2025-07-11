# pde_wandb_log.py
import os, numpy as np
from einops import rearrange
from PDE.model.reaction_diffusion_advection.visualize import (
        plot_weight_kernel_boxplot, plot_weight_matrices)
from Common.trainer.abstract_wandb_log import Train_log   # ⬅️ no TF!

class PDE_Train_log(Train_log):
    """
    Logging helper for PDE_Trainer – pure wandb/tensorboard backend
    (whatever Train_log is configured for), zero TensorFlow deps.
    """

    def __init__(self, log_dir: str, data, RGB_mode="RGB"):
        super().__init__(save_dir=log_dir)        # initialises backend
        self.RGB_mode = RGB_mode

        # ––– log target sequence once (step = i) ––––––––––––––
        for i in range(len(data[0])):
            frames = np.stack([im[i, :3] for im in data])          # [B,C,X,Y]
            self.log_image("True sequence", rearrange(frames, "b c x y -> b x y c"),
                           step=i)

    # ---------- per-iteration hooks ----------------------------
    def log_model_parameters(self, model, i: int):
        figs = plot_weight_matrices(model)
        self.log_image("Weight matrices", np.array(figs)[:, 0], step=i)

        figs = plot_weight_kernel_boxplot(model)
        self.log_image("Input weights per channel", np.array(figs)[:, 0], step=i)

    def log_model_outputs(self, x, i: int):
        """x : list[ BATCH ][ SUBSTEP × C × X × Y ]"""
        B = len(x)
        last_rgb = rearrange(x, "b n c x y -> b n x y c")[:, -1, :, :, :3]
        self.log_image("Training outputs", last_rgb, step=i)

        if x[0].shape[1] > 4:
            hid = []
            for b in range(B):
                h = x[b][-1, 3:]
                pad = (-h.shape[0]) % 3
                hid.append(np.pad(h, ((0, pad), (0, 0), (0, 0))))
            hid = rearrange(hid, "b (z c) x y -> b (z x) y c", c=3)
            self.log_image("Training hidden channels", hid, step=i)

    # optional end-of-training video
    def tb_training_end_log(self, pde, x, ts, boundary_callback, write_images=True):
        if not write_images:
            return
        _, Y = pde(ts, x[0][0])        # one batch for illustration
        rgb = rearrange(Y[:, :3], "t c x y -> t x y c")
        self.log_video("Final PDE trajectory", rgb, step=None)

