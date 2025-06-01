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
TIME_SAMPLING = 32      # NCA micro-steps between PDE snapshots
NUM_INTERVALS = 8       # number of snapshot intervals
SIZE          = 64
BATCHES       = 1       # single trajectory, as requested

# Schnakenberg parameters (same as training)
a, b, D = 0.2, 0.8, 50.0
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
key = jr.PRNGKey(0)

# initial condition: equilibrium + small uniform noise  (shape [B,C,H,W])
U0 = (U_eq + noise_amp * jr.uniform(key,  (BATCHES, 1, SIZE, SIZE)))
V0 = (V_eq + noise_amp * jr.uniform(key,  (BATCHES, 1, SIZE, SIZE)))
x0 = jnp.concatenate([U0, V0], axis=1)                  # [B,2,H,W]

# Smooth the IC the same way as training (3× 3×3 mean filter)
for _ in range(3):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# Integrate PDE
ts = jnp.linspace(0, TIME_SAMPLING * NUM_INTERVALS, TIME_SAMPLING * NUM_INTERVALS)
_, Y = solver(ts, x0)                                   # Y: [T,B,2,H,W]
Y = rearrange(Y, "T B C X Y -> B T C X Y")              # [B,T,C,H,W]
Y = Y[:, :, :1]                                         # keep U-channel only

# Global normalisation (identical to training)
mn, mx = Y.min(), Y.max()
Y = (Y - mn) / (mx - mn)
Y = Y[:, ::TIME_SAMPLING]                               # down-sample in time
assert Y.shape[1] == NUM_INTERVALS                      # [B,NUM_INTERVALS,1,H,W]

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
    key=jr.PRNGKey(0),
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
        k, sub = jr.split(k)
        x = nca(x, lambda y: y, sub)                    # identical call-signature
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
