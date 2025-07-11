# visualize_pde.py     (TensorFlow-free)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange
from PIL import Image
import io
import jax

# ---------- helpers -------------------------------------------------
def plot_to_image(figure):
    """
    Convert a Matplotlib figure to a numpy image with shape (1, H, W, C),
    so Train_log.log_image() can consume it.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    # read PNG bytes with Pillow instead of tf.image.decode_png
    image = Image.open(buf)          # RGB(A) → numpy
    image = np.array(image)

    # add batch dimension to match the (1, H, W, C) convention
    image = rearrange(image, "h w c -> () h w c")
    return image

# ---------- weight-matrix heatmaps ----------------------------------
def plot_weight_matrices(pde):
    """
    Produce a list of images (numpy arrays) that visualise every dense layer
    in advection / diffusion / reaction sub-nets.
    """
    figs = []

    # collect weights in exactly the same way as before
    w_v, w_d, w_r, w_r_p, w_r_d = [], [], [], [], []
    for i in range(pde.func.N_LAYERS + 1):
        if "advection"          in pde.func.TERMS: w_v.append(pde.func.f_v.layers[2*i].weight[:,:,0,0])
        if "diffusion_nonlinear" in pde.func.TERMS: w_d.append(pde.func.f_d.layers[2*i].weight[:,:,0,0])
        if "diffusion"          in pde.func.TERMS: w_d.append(pde.func.f_d.layers[2*i].weight[:,:,0,0])
        if "reaction_split"     in pde.func.TERMS:
            w_r_p.append(pde.func.f_r.production_layers[2*i].weight[:,:,0,0])
            w_r_d.append(pde.func.f_r.decay_layers[2*i].weight[:,:,0,0])
        if "reaction_pure"      in pde.func.TERMS: w_r.append(pde.func.f_r.layers[2*i].weight[:,:,0,0])

    # helper to render a single heat-map
    def _add_heatmaps(ws, title_prefix):
        for idx, w in enumerate(ws):
            fig = plt.figure(figsize=(5, 5))
            col = max(np.max(w), -np.min(w))
            plt.imshow(w, cmap="seismic", vmax=col, vmin=-col)
            plt.ylabel("Output"), plt.xlabel("Input")
            plt.title(f"{title_prefix} layer {idx}")
            figs.append(plot_to_image(fig))

    if "advection"          in pde.func.TERMS: _add_heatmaps(w_v, "Advection")
    if "diffusion_nonlinear" in pde.func.TERMS: _add_heatmaps(w_d, "Diffusion")
    if "diffusion"          in pde.func.TERMS: _add_heatmaps(w_d, "Nonlinear Diffusion")
    if "reaction_split"     in pde.func.TERMS:
        _add_heatmaps(w_r_p, "Reaction production")
        _add_heatmaps(w_r_d, "Reaction decay")
    if "reaction_pure"      in pde.func.TERMS: _add_heatmaps(w_r, "Reaction")

    return figs

# ---------- 1st-layer box-plots ------------------------------------
def plot_weight_kernel_boxplot(pde):
    """
    Box-plots of the very first layer in each sub-net, split by channel.
    """
    figs = []

    def _boxplot_rows(w, title):
        fig = plt.figure(figsize=(5, 5))
        plt.boxplot(w.T)
        plt.xlabel("Channels"), plt.ylabel("Weights"), plt.title(title)
        figs.append(plot_to_image(fig))

    if "advection" in pde.func.TERMS:
        _boxplot_rows(pde.func.f_v.layers[0].weight[:,:,0,0], "Advection 1st layer")

    if "diffusion_nonlinear" in pde.func.TERMS:
        _boxplot_rows(pde.func.f_d.layers[0].weight[:,:,0,0], "Diffusion 1st layer")

    if "diffusion_linear" in pde.func.TERMS:
        w = pde.func.f_d.diffusion_constants[:,0,0]
        fig = plt.figure(figsize=(5,5))
        plt.bar(np.arange(len(w)), jax.nn.sparse_plus(w))
        plt.xlabel("Channels"), plt.ylabel("Weights"), plt.title("Diffusion coefficients")
        figs.append(plot_to_image(fig))

    if "diffusion" in pde.func.TERMS:
        _boxplot_rows(pde.func.f_d.layers[0].weight[:,:,0,0], "Non-linear Diffusion 1st layer")
        w_lin = pde.func.f_d.diffusion_constants[:,0,0]
        fig = plt.figure(figsize=(5,5))
        plt.bar(np.arange(len(w_lin)), jax.nn.sparse_plus(w_lin))
        plt.xlabel("Channels"), plt.ylabel("Weights"), plt.title("Linear Diffusion coefficients")
        figs.append(plot_to_image(fig))

    if "reaction_split" in pde.func.TERMS:
        _boxplot_rows(pde.func.f_r.production_layers[0].weight[:,:,0,0], "Reaction production 1st layer")
        _boxplot_rows(pde.func.f_r.decay_layers[0].weight[:,:,0,0], "Reaction decay 1st layer")

    if "reaction_pure" in pde.func.TERMS:
        _boxplot_rows(pde.func.f_r.layers[0].weight[:,:,0,0], "Reaction 1st layer")

    return figs

# ---------- optional animation util (unchanged) --------------------
def my_animate(img, clip=True):
    """
    Produce an interactive Matplotlib animation from an array of frames.
    img : float[ N, rgb, H, W ] in [0,1] or int
    """
    if clip:  img = np.clip(img, 0.0, 1.0)
    img = np.einsum("ncxy->nxyc", img)

    frames, fig = [], plt.figure()
    for t in range(img.shape[0]):
        frames.append([plt.imshow(img[t], animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=0)
    plt.show()

# ---------- helper to sort kernel strings (unchanged) --------------
def sort_kstr(K_STR):
    K_SORTED = []
    for s in ["ID", "DIFF", "GRAD", "AV", "LAP"]:
        if s in K_STR: K_SORTED.append(s)
    return K_SORTED
