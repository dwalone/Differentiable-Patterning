#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 0. Imports & CLI
# -----------------------------------------------------------------------------
import argparse, os, sys, time, functools
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from einops import rearrange

sys.path.append("..")                                    # repo root
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA                      # same class used in training

# -----------------------------------------------------------------------------
# 1. Hyper-parameters (identical to training)
# -----------------------------------------------------------------------------
CHANNELS      = 8
KERNEL_STR    = ["ID", "LAP", "GRAD"]
FIRE_RATE     = 1.0
PADDING       = "CIRCULAR"
KERNEL_SCALE  = 1
DX            = 1.0
DT_MICRO      = 5e-3
TIME_SAMPLING = 64      # NCA micro-steps between PDE snapshots
NUM_INTERVALS = 10       # number of snapshot intervals
SIZE          = 64
BATCHES       = 1       # single trajectory, as requested
STATE_REGULARISER    = 1.0
FIRE_RATE     = 1.0

# Schnakenberg parameters (same as training)
a, b, D = 0.01, 2, 80
U_eq = a + b
V_eq = b / (U_eq**2)
noise_amp = 0.05

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
# 3. Build PDE RHS & solver – exactly as in training
# -----------------------------------------------------------------------------
op    = Ops(PADDING=PADDING, dx=DX, KERNEL_SCALE=3)
f_rhs = F_schnakenberg(PADDING=PADDING, dx=DX, KERNEL_SCALE=1,
                       a=a, b=b, D=D)
v_rhs = eqx.filter_vmap(f_rhs, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(v_rhs, DT_MICRO)

# -----------------------------------------------------------------------------
# 4. Generate ground-truth trajectory
# -----------------------------------------------------------------------------
key = jr.PRNGKey(100)


# gaussian noise ##################
sigma = 0.03          # noise amplitude  (3 % of steady state)
key, sub1, sub2 = jr.split(key, 3)

# sample independent noise for each batch item
U0 = U_eq + sigma * jr.normal(sub1, shape=(BATCHES, 1, SIZE, SIZE))
V0 = V_eq + sigma * jr.normal(sub2, shape=(BATCHES, 1, SIZE, SIZE))

x0 = jnp.concatenate([U0, V0], axis=1)   # shape (B, 2, SIZE, SIZE)
####################################

# single-dot
# homogeneous steady state everywhere …
# U0 = jnp.full((BATCHES, 1, SIZE, SIZE), U_eq)
# V0 = jnp.full((BATCHES, 1, SIZE, SIZE), V_eq)
# # … except one activator peak at the lattice centre
# cx, cy = SIZE // 2, SIZE // 2
# U0 = U0.at[:, 0, cx, cy].set(U_eq + 0.2)   # 0.2 ≈ 10 % bump, adjust as needed
# x0 = jnp.concatenate([U0, V0], axis=1)     # shape [B, 2, SIZE, SIZE]
######################################################

# identical to training
sampling_constant = 32
ts = jnp.linspace(0, sampling_constant * 3, sampling_constant * NUM_INTERVALS)
T, Y = solver(ts, x0)
# Y has shape [T, B, 2, SIZE, SIZE]

# reshape and keep only U-channel for NCA training
Y = rearrange(Y, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                                 # drop V
Y = (Y - Y.min()) / (Y.max() - Y.min())         # normalize [0,1]
Y = Y[:, ::sampling_constant]                       # downsample in time

# -----------------------------------------------------------------------------
# 5. Load trained NCA
# -----------------------------------------------------------------------------
dummy_nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE,
    KERNEL_SCALE=KERNEL_SCALE,
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
hidden_shape  = (BATCHES, CHANNELS - 1, SIZE, SIZE)
x_nca0 = jnp.concatenate([Y[:, 0], jnp.zeros(hidden_shape)], axis=1)  # [B,8,H,W]

# Roll through all NUM_INTERVALS snapshots
nca_preds = []
x_cur = x_nca0
k_batch = jr.split(key, BATCHES)                        # per-batch RNG keys
for step in range(NUM_INTERVALS):
    nca_preds.append(x_cur[:, :1])                      # store U channel
    x_cur = v_rollout(x_cur, k_batch)                  # 32 micro-steps

nca_preds = jnp.stack(nca_preds, axis=1)                # [B,T,1,H,W]

# Apply the same normalisation as GT
#nca_preds = (nca_preds - mn) / (mx - mn)

# -----------------------------------------------------------------------------
# 7. Loss – same Euclidean metric as training
# -----------------------------------------------------------------------------
loss = jnp.mean((nca_preds - Y) ** 2)
print(f"Normalised Euclidean loss over the trajectory: {float(loss):.6e}")

# # -----------------------------------------------------------------------------
# # 8. Save trajectory images: GT • NCA • error
# # -----------------------------------------------------------------------------
# for t in range(NUM_INTERVALS):
#     gt   = Y[0, t, 0]
#     pred = nca_preds[0, t, 0]
#     err  = (pred - gt)**2
#     print(jnp.mean(err))
#     fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
#     ax[0].imshow(gt,   origin="lower"); ax[0].set_title("Ground truth U")
#     ax[1].imshow(pred, origin="lower"); ax[1].set_title("NCA U")
#     im = ax[2].imshow(err, origin="lower"); ax[2].set_title("Error")
#     fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
#     for a in ax: a.axis("off")
#     fig.suptitle(f"Snapshot {t} (after {t*TIME_SAMPLING} micro-steps)")
#     fname = os.path.join(args.outdir, f"step_{t:02d}.png")
#     fig.savefig(fname, dpi=150)
#     plt.close(fig)

# print(f"Images written to: {os.path.abspath(args.outdir)}")

# -------------------------------------------------------------------------
# 8. Save a single 3×T figure: GT • NCA • error
# -------------------------------------------------------------------------
fig, axes = plt.subplots(
    3, NUM_INTERVALS, figsize=(3 * NUM_INTERVALS, 9), constrained_layout=True
)

for t in range(NUM_INTERVALS):
    gt   = Y[0, t, 0]
    pred = nca_preds[0, t, 0]
    err  = (pred - gt) ** 2

    # first row – ground truth
    ax = axes[0, t]
    ax.imshow(gt, origin="lower", cmap="viridis")
    ax.set_title(f"GT t={t}")
    ax.axis("off")

    # second row – prediction
    ax = axes[1, t]
    ax.imshow(pred, origin="lower", cmap="viridis")
    ax.set_title(f"Pred t={t}")
    ax.axis("off")

    # third row – error heat-map
    ax = axes[2, t]
    im = ax.imshow(err, origin="lower", cmap="inferno")
    ax.set_title(f"Err t={t}")
    ax.axis("off")

# colour-bar for error maps
cbar = fig.colorbar(im, ax=axes[2, :], orientation="horizontal", shrink=0.7)
cbar.set_label("Squared error")

fig.suptitle(
    f"Schnakenberg trajectory: GT vs. NCA prediction — "
    f"loss={float(loss):.3e}", fontsize=14
)

out_path = os.path.join(args.outdir, "trajectory_grid.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)

print(f"Grid image written to: {os.path.abspath(out_path)}")

