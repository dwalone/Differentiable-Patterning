#!/usr/bin/env python
# ------------------------------------------------------------------
# 0. Imports & CLI
# ------------------------------------------------------------------
import argparse, os, sys
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from einops import rearrange

sys.path.append("..")                                 # repo root
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_gierer_meinhardt import (
    F as F_gm,                                         # ← CHANGE
)
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA

# ------------------------------------------------------------------
# 1. Hyper-parameters (same as training)
# ------------------------------------------------------------------
CHANNELS      = 8
KERNEL_STR    = ["ID", "LAP", "GRAD"]
FIRE_RATE     = 1.0
PADDING       = "CIRCULAR"
KERNEL_SCALE  = 1
DX            = 1.0
DT_MICRO      = 1e-3           # ← CHANGE (matches training script)
TIME_SAMPLING = 32
NUM_INTERVALS = 8
SIZE          = 64
BATCHES       = 1              # one trajectory

# Gierer–Meinhardt parameters
D, a, b, c = 100.0, 0.5, 1.0, 6.1

# ------------------------------------------------------------------
# 2. CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True,
                    help="Path to trained *.eqx checkpoint")
parser.add_argument("--outdir", default="inference_results_gm",
                    help="Directory for PNGs")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# ------------------------------------------------------------------
# 3. Build RHS & solver
# ------------------------------------------------------------------
op    = Ops(PADDING=PADDING, dx=DX, KERNEL_SCALE=3)
f_rhs = F_gm(PADDING=PADDING, dx=DX, a=a, b=b, c=c, D=D)
v_rhs = eqx.filter_vmap(f_rhs, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(v_rhs, DT_MICRO)

# ------------------------------------------------------------------
# 4. Ground-truth trajectory (two-dot IC)
# ------------------------------------------------------------------
key = jr.PRNGKey(0)

u0 = jnp.ones((BATCHES, 1, SIZE, SIZE))
v0 = jnp.ones((BATCHES, 1, SIZE, SIZE))

# two Gaussian bumps in u
yy, xx = jnp.ogrid[:SIZE, :SIZE]
dots = [(SIZE//4, SIZE//4), (3*SIZE//4, 3*SIZE//4)]
for cx, cy in dots:
    gauss = jnp.exp(-((xx-cx)**2 + (yy-cy)**2) / 4.0)
    u0 += 0.30 * gauss[None, None]

x0 = jnp.concatenate([u0, v0], axis=1)

# same 3× blur as training
for _ in range(3):
    x0 = jax.vmap(op.Average, 0, 0)(x0)

# integrate PDE
ts = jnp.linspace(0, TIME_SAMPLING * NUM_INTERVALS,
                  TIME_SAMPLING * NUM_INTERVALS)
_, Y = solver(ts, x0)                               # [T,B,2,H,W]
Y = rearrange(Y, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                                     # observe u only

mn, mx = Y.min(), Y.max()
ptp = jnp.where(mx - mn < 1e-8, 1.0, mx - mn)       # avoid 0 divisor
Y = (Y - mn) / ptp
Y = Y[:, ::TIME_SAMPLING]                           # [B,8,1,H,W]

# ------------------------------------------------------------------
# 5. Load trained NCA
# ------------------------------------------------------------------
dummy = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE,
    KERNEL_SCALE=KERNEL_SCALE,
    key=key,
)
nca = eqx.tree_deserialise_leaves(args.model, dummy)

# ------------------------------------------------------------------
# 6. NCA rollout helper
# ------------------------------------------------------------------
def rollout_single(state, rng):
    def step_fn(carry, _):
        x, k = carry
        k = jr.fold_in(k, 0)
        x = nca(x, lambda y: y, k)
        return (x, k), x
    (_, _), xs = jax.lax.scan(step_fn, (state, rng), None,
                              length=TIME_SAMPLING)
    return xs[-1]

v_rollout = jax.vmap(rollout_single, 0, 0)

# lattice with hidden channels zero
hidden = jnp.zeros((BATCHES, CHANNELS-1, SIZE, SIZE))
x_cur  = jnp.concatenate([Y[:, 0], hidden], axis=1)

nca_preds = []
k_batch   = jr.split(key, BATCHES)
for _ in range(NUM_INTERVALS):
    nca_preds.append(x_cur[:, :1])
    x_cur = v_rollout(x_cur, k_batch)

nca_preds = jnp.stack(nca_preds, axis=1)            # [B,T,1,H,W]

# ------------------------------------------------------------------
# 7. Loss
# ------------------------------------------------------------------
loss = jnp.mean((nca_preds - Y) ** 2)
print(f"Normalised Euclidean loss: {float(loss):.6e}")

# ------------------------------------------------------------------
# 8. Save comparison images
# ------------------------------------------------------------------
for t in range(NUM_INTERVALS):
    gt   = Y[0, t, 0]
    pred = nca_preds[0, t, 0]
    err  = (pred - gt) ** 2
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    ax[0].imshow(gt,   origin="lower"); ax[0].set_title("GT u")
    ax[1].imshow(pred, origin="lower"); ax[1].set_title("NCA u")
    im = ax[2].imshow(err, origin="lower"); ax[2].set_title("Squared error")
    for a in ax: a.axis("off")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
    fig.suptitle(f"Snapshot {t} (after {t*TIME_SAMPLING} micro-steps)")
    fig.savefig(os.path.join(args.outdir, f"step_{t:02d}.png"), dpi=150)
    plt.close(fig)

print("Images written to", os.path.abspath(args.outdir))
