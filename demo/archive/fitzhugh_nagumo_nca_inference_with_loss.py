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

sys.path.append("..")                                    # repo root
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_fhn import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA                      # same class used in training

# -----------------------------------------------------------------------------
# 1. Hyper-parameters (identical to training)
# -----------------------------------------------------------------------------
CHANNELS      = 16
KERNEL_STR    = ["ID", "LAP", "GRAD"]
FIRE_RATE     = 1.0
PADDING       = "CIRCULAR"
DX            = 1.0
DT_MICRO      = 5e-5
TIME_SAMPLING = 32      # NCA micro-steps between PDE snapshots
NUM_INTERVALS = 8       # number of snapshot intervals
SIZE          = 64
BATCHES       = 2

# FHN parameters
D_true    = 20         # slower inhibitor diffusion
eps_v_true = 0.5       # stronger timescale separation
a_v_true   = 1
a_z_true   = -0.1         # zero offset → excitable pulses

# domain and discretization
dx = 1.0
dt = 1e-2

def make_spike_ic(key, B, H, W, N_clicks=5, sigma=1.5, amplitude=1.0):
    """
    Returns array [B,2,H,W] in float64 with N_clicks Gaussian bumps in channel 0.
    Channel 1 is zero.
    """
    # 1) precompute Gaussian kernel in float64
    radius = int(3 * sigma)
    xs = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    ys = xs
    Xg, Yg = jnp.meshgrid(xs, ys, indexing='ij')
    kernel = amplitude * jnp.exp(-(Xg**2 + Yg**2) / (2 * sigma**2))

    # 2) init U,V in float64
    U0 = jnp.zeros((B, H, W), dtype=jnp.float64)
    V0 = jnp.zeros_like(U0)

    # 3) get keys
    keys = jr.split(key, B * N_clicks).reshape(B, N_clicks, 2)

    # 4) scatter‐add bumps
    for b in range(B):
        for n in range(N_clicks):
            subkey = keys[b, n]
            k1, k2 = jr.split(subkey)
            i = int(jr.randint(k1, (), radius, H - radius))
            j = int(jr.randint(k2, (), radius, W - radius))
            
            i0, i1 = i - radius, i + radius + 1
            j0, j1 = j - radius, j + radius + 1
            U0 = U0.at[b, i0:i1, j0:j1].add(kernel)

    # 5) stack u,v channels
    return jnp.stack([U0, V0], axis=1)  # shape [B,2,H,W] dtype=float64

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
func  = F_fhn(
    PADDING="CIRCULAR",
    dx=dx,
    D=D_true,
    eps_v=eps_v_true,
    a_v=a_v_true,
    a_z=a_z_true
)
# wrap fhn so it handles batch axis
vfunc = eqx.filter_vmap(func, in_axes=(None,0,None), out_axes=0)
solver = PDE_solver(vfunc, dt)

# -----------------------------------------------------------------------------
# 4. Generate ground-truth trajectory
# -----------------------------------------------------------------------------
key = jr.PRNGKey(4)

# 3) IC: random clicks, already float64
key, subkey = jr.split(key)
x0 = make_spike_ic(
    subkey,
    B=BATCHES,
    H=SIZE,
    W=SIZE,
    N_clicks=10,
    sigma=1.0,
    amplitude=1.0
)   # [B,2,H,W] float64

# 4) smooth in float64
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
# op kernels are float64, x0 is float64 → no dtype mismatches
for _ in range(2):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# Integrate PDE
ts = jnp.linspace(0.0, TIME_SAMPLING * 8 * 0.1, TIME_SAMPLING * 8, dtype=jnp.float64)
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
