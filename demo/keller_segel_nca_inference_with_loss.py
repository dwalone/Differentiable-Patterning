#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 0. Imports & CLI
# -----------------------------------------------------------------------------
import argparse, os, sys, time, functools
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from einops import rearrange
from einops import reduce

sys.path.append("..")                             
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_keller_segel import F as F_keller_segel
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA

# -----------------------------------------------------------------------------
# 1. Hyper-parameters (identical to training)
# -----------------------------------------------------------------------------
CHANNELS      = 16
KERNEL_STR    = ["ID", "LAP", "GRAD"]
FIRE_RATE     = 1.0
PADDING       = "CIRCULAR"
TIME_SAMPLING = 32    
NUM_INTERVALS = 16      
SIZE          = 64
BATCHES       = 6


CELL_CHANNELS   = 1
SIGNAL_CHANNELS = 1
dx              = 0.5
dt_true         = 0.1
SOLVER_PARAMS   = {
    "dt": dt_true,
    "SOLVER": "heun",
    "rtol": 1e-3,
    "atol": 1e-3,
    "ADAPTIVE": True,
    "DTYPE": "float32"
}

# Keller–Segel coefficients
alpha   = 0.01
c       = 3.8
D       = 0.8
epsilon = 0.1

# Trajectory parameters
T_FINAL          = 200.0
TIME_RESOLUTION  = 512   # number of saved frames between t=0 and t=200
TIME_SAMPLING    = 32


def make_random_ic(key, B, H, W, CELL_CHANNELS, SIGNAL_CHANNELS):
    """
    Returns an array of shape [B, C_total, H, W], dtype=float32:
      - channel 0 (cell) ∼ Uniform(0, 0.1)
      - channel 1 (signal) = 0
    """
    key, subkey = jr.split(key)
    X = jr.uniform(
        subkey,
        shape=(B, CELL_CHANNELS + SIGNAL_CHANNELS, H, W),
        minval=0.0,
        maxval=0.1,
        dtype=jnp.float32
    )
    # Zero out the signal channel
    X = X.at[:, CELL_CHANNELS:].set(0.0)
    return X

# -----------------------------------------------------------------------------
# 2. Command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",  type=str, required=True,
                    help="Path to the trained *.eqx checkpoint")
parser.add_argument("--outdir", type=str, default="inference_results",
                    help="Directory to store trajectory PNGs")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. Build PDE RHS & solver
# -----------------------------------------------------------------------------
func = F_keller_segel(
    PADDING="CIRCULAR",
    dx=dx,
    KERNEL_SCALE=1,
    alpha=alpha,
    c=c,
    D=D,
    epsilon=epsilon
)
# wrap fhn so it handles batch axis
vfunc = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, **SOLVER_PARAMS)

# -----------------------------------------------------------------------------
# 4. Generate ground-truth trajectory
# -----------------------------------------------------------------------------
key = jr.PRNGKey(1)

# 3) IC: random clicks, already float64
key, subkey = jr.split(key)
x0 = make_random_ic(
    subkey,
    B=BATCHES,
    H=SIZE,
    W=SIZE,
    CELL_CHANNELS=CELL_CHANNELS,
    SIGNAL_CHANNELS=SIGNAL_CHANNELS
)  # shape [B, 2, 64, 64], float32

# One pass of circular averaging to smooth out high-frequency noise
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=2)
v_av = eqx.filter_vmap(op.Average, in_axes=0, out_axes=0)
x0 = v_av(x0)  # shape still [B, 2, 64, 64]

# Integrate PDE
ts = jnp.linspace(0.0, T_FINAL, TIME_RESOLUTION, dtype=jnp.float32)
_, Y_full = solver(ts, x0)                                   # Y: [T,B,2,H,W]
Y_full = rearrange(Y_full, "T B C X Y -> B T C X Y")
for ch in range(CELL_CHANNELS + SIGNAL_CHANNELS):
    ch_min = Y_full[:, :, ch].min()
    ch_max = Y_full[:, :, ch].max()
    Y_full = Y_full.at[:, :, ch].set((Y_full[:, :, ch] - ch_min) / (ch_max - ch_min))
Y = Y_full[:, ::TIME_SAMPLING, :, :, :]

# -----------------------------------------------------------------------------
# 5. Load trained NCA
# -----------------------------------------------------------------------------
dummy_nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=FIRE_RATE,
    key=key,
)
nca = eqx.tree_deserialise_leaves(args.model, dummy_nca)

# -----------------------------------------------------------------------------
# 6. Roll the NCA forward with the same parallelisation pattern
# -----------------------------------------------------------------------------
def rollout_single(initial_state, rng):
    """
    Rollout TIME_SAMPLING micro-steps and return the final lattice state.
    Mirrors the scan-based stepping used in NCA_Trainer.compute_loss.
    """
    def step_fn(carry, _):
        x, k = carry
        k = jr.fold_in(k, 0)
        subkey = k
        x = nca(x, lambda y: y, subkey)                    # identical call-signature
        return (x, k), x

    (_, _), xs = jax.lax.scan(step_fn, (initial_state, rng), None,
                              length=TIME_SAMPLING)
    return xs[-1]

# vectorise across batch (even though B=1 – keeps code identical to training)
v_rollout = jax.vmap(rollout_single, in_axes=(0, 0), out_axes=0)

# Prepare NCA lattice: observed channel(s) in front, hidden-state zeros elsewhere
hidden_shape  = (BATCHES, CHANNELS - 2, SIZE, SIZE)
x_nca0 = jnp.concatenate([Y[:, 0, :2], jnp.zeros(hidden_shape)], axis=1)  # [B,8,H,W]

# Roll through all NUM_INTERVALS snapshots
nca_preds = []
x_cur = x_nca0
k_batch = jr.split(key, BATCHES)              
for step in range(NUM_INTERVALS):
    nca_preds.append(x_cur[:, :2])                     
    x_cur = v_rollout(x_cur, k_batch)  

nca_preds = jnp.stack(nca_preds, axis=1)                # [B,T,1,H,W]

# Apply the same normalisation as GT
#nca_preds = (nca_preds - mn) / (mx - mn)


# ----------------------------------------------------------------------------- 
# 7. Loss – same Euclidean metric as training
# ----------------------------------------------------------------------------- 
loss = jnp.mean((nca_preds - Y) ** 2)
print(f"Normalised Euclidean loss over the trajectory: {float(loss):.6e}")
# # ----------------------------------------------------------------------------- 
# # 7. Multi-scale loss (scales 1,2,4 – exactly like the trainer)
# # ----------------------------------------------------------------------------- 
# LOSS_SCALES = [1, 2, 4]

# def _downsample(arr, d):
#     """Mean-pool by factor d in both spatial dims using einops.reduce"""
#     if d == 1:
#         return arr
#     return reduce(
#         arr,
#         "B T C (h dh) (w dw) -> B T C h w",
#         "mean",
#         dh=d,
#         dw=d,
#     )

# ms_losses = []
# for d in LOSS_SCALES:
#     Xd = _downsample(nca_preds, d)
#     Yd = _downsample(Y,         d)
#     ms_losses.append(jnp.mean((Xd - Yd) ** 2))

# loss = jnp.mean(jnp.stack(ms_losses))
# print(
#     "Multi-scale Euclidean loss  "
#     + "/".join(f"d{d}:{l:.4f}" for d, l in zip(LOSS_SCALES, ms_losses))
#     + f"   ➜   mean: {float(loss):.4f}"
# )


# -----------------------------------------------------------------------------
# 8. Save trajectory images: GT • NCA • error
# -----------------------------------------------------------------------------
for t in range(NUM_INTERVALS):
    gt   = Y[0, t, 0]
    pred = nca_preds[0, t, 0]
    err  = (pred - gt)**2
    print(jnp.mean(err))
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    ax[0].imshow(gt,   origin="lower"); ax[0].set_title("Ground truth U")
    ax[1].imshow(pred, origin="lower"); ax[1].set_title("NCA U")
    im = ax[2].imshow(err, origin="lower"); ax[2].set_title("Error")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
    for a in ax: a.axis("off")
    fig.suptitle(f"Snapshot {t} (after {t*TIME_SAMPLING} micro-steps)")
    fname = os.path.join(args.outdir, f"step_{t:02d}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)

print(f"Images written to: {os.path.abspath(args.outdir)}")
