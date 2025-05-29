#!/usr/bin/env python
import os, sys
import jax, jax.random as jr, jax.numpy as jnp
import equinox as eqx
from einops import rearrange

# ─── adjust to point at your repo root ───
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
import Common.trainer.loss as loss          # your loss module

# 1) Hyper‐parameters (must match training)
CHANNELS      = 8
KERNEL_STR    = ["ID","LAP","GRAD"]
FIRE_RATE     = 1.0
PADDING       = "CIRCULAR"
KERNEL_SCALE  = 1
DX            = 1.0
DT_MICRO      = 5e-3
TIME_SAMPLING = 32
NUM_INTERVALS = 8
SIZE          = 64
BATCHES       = 1   # single trajectory

# fixed PDE parameters
a, b, D = 0.2, 0.8, 50.0
U_eq = a + b
V_eq = b / (U_eq**2)
noise = 0.05

# Pre‐build the spatial op + PDE solver
op = Ops(PADDING=PADDING, dx=DX, KERNEL_SCALE=3)
func   = F_schnakenberg(PADDING=PADDING, dx=DX, KERNEL_SCALE=1,
                        a=a, b=b, D=D)
vfunc  = eqx.filter_vmap(func, in_axes=(None,0,None), out_axes=0)
solver = PDE_solver(vfunc, DT_MICRO)

# 2) Load trained NCA
MODEL_FILE = "models/demo/train_nca_to_pde_schnakenberg.eqx"
dummy_nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE,
    KERNEL_SCALE=KERNEL_SCALE,
    key=jr.PRNGKey(0),
)
nca = eqx.tree_deserialise_leaves(MODEL_FILE, dummy_nca)

# 3) Roll‐out function (constant dt=1 per micro‐step)
def rollout_nca(nca, x0, n_steps, n_intervals, key):
    traj = []
    x = x0
    for _ in range(n_intervals):
        traj.append(x)
        for _ in range(n_steps):
            key, subk = jr.split(key)
            x = nca(x, boundary_callback=lambda z: z, key=subk)
    return jnp.stack(traj, axis=0)  # [K, C, H, W]

# Prepare a zero‐pad channel tensor
hidden = jnp.zeros((CHANNELS-2, SIZE, SIZE), dtype=jnp.float32)

# 4) Instantiate a “dummy” trainer for just computing loss
trainer = NCA_Trainer(
    nca,
    data=jnp.zeros((1, NUM_INTERVALS, CHANNELS, SIZE, SIZE)),  # placeholder
    model_filename=None,
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True,
    OBS_CHANNELS=2
)
# Patch in the same loss you used in training:
trainer._loss_func = loss.euclidean

def one_validation_run(seed: int):
    # A) sample a fresh noisy PDE IC
    key = jr.PRNGKey(seed)
    key, k1 = jr.split(key)
    U0 = U_eq + noise * jr.normal(k1, (BATCHES,1,SIZE,SIZE))
    key, k2 = jr.split(key)
    V0 = V_eq + noise * jr.normal(k2, (BATCHES,1,SIZE,SIZE))
    x0 = jnp.concatenate([U0, V0], axis=1)
    # same smoothing
    for _ in range(3):
        x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

    # B) re‐solve the PDE and downsample
    T_steps = TIME_SAMPLING * NUM_INTERVALS
    ts = jnp.linspace(0.0, float(T_steps), T_steps)
    _, Y_full = solver(ts=ts, y0=x0)
    Y = rearrange(Y_full, "T B C X Y -> B T C X Y")[0]     # [K,2,H,W]
    Y_snap = Y[::TIME_SAMPLING]                           # [K,2,H,W]

    # C) roll out the NCA
    init_state = jnp.concatenate([Y_snap[0], hidden], axis=0)  # [C,H,W]
    X_pred = rollout_nca(nca, init_state, TIME_SAMPLING, NUM_INTERVALS, key)

    # Now compute loss for all intervals 0→1, 1→2, …, 7→8:
    interval_losses = []
    for i in range(NUM_INTERVALS):
        x_i   = X_pred[i][None,...]                                      # [1,C,H,W]
        y_i   = jnp.concatenate([Y_snap[i+1], hidden], axis=0)[None,...] # [1,C,H,W]
        key_i = jr.fold_in(jr.PRNGKey(seed), i)
        l     = trainer.loss_func(x_i, y_i, key_i)  # shape [1]
        interval_losses.append(l[0])
    return jnp.stack(interval_losses)  # shape [NUM_INTERVALS]

seeds = list(range(10))
seeds = [x * 5 for x in seeds]
# This now gives a list of 10 arrays, each of shape [8]
all_losses = jnp.stack([one_validation_run(s) for s in seeds], axis=0)
# all_losses.shape == (10, 8)

# Mean/std _per interval_ across the 10 trials:
mean_per_interval = all_losses.mean(axis=0)  # shape [8]
std_per_interval  = all_losses.std(axis=0)   # shape [8]

print("Per-interval loss (mean ± std):")
for i,(m,s) in enumerate(zip(mean_per_interval, std_per_interval)):
    print(f"  Interval {i}→{i+1}: {m:.5f} ± {s:.5f}")

# Overall across _all_ intervals and seeds:
overall_mean = all_losses.mean()
overall_std  = all_losses.std()
print(f"\nOverall mean ± std = {overall_mean:.5e} ± {overall_std:.5e}")

