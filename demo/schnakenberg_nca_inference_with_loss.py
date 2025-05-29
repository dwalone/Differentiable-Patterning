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
import Common.trainer.loss as loss          # <— import your loss module

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

# 2) Build “ground-truth” PDE data
key = jr.PRNGKey(0)
a, b, D = 0.2, 0.8, 50.0
U_eq = a + b
V_eq = b / (U_eq**2)
noise = 0.05

key, k1 = jr.split(key)
U0 = U_eq + noise * jr.normal(k1, (BATCHES,1,SIZE,SIZE))
key, k2 = jr.split(key)
V0 = V_eq + noise * jr.normal(k2, (BATCHES,1,SIZE,SIZE))
x0 = jnp.concatenate([U0, V0], axis=1)

op = Ops(PADDING=PADDING, dx=DX, KERNEL_SCALE=3)
for _ in range(3):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

func   = F_schnakenberg(PADDING=PADDING, dx=DX, KERNEL_SCALE=1,
                        a=a, b=b, D=D)
vfunc  = eqx.filter_vmap(func, in_axes=(None,0,None), out_axes=0)
solver = PDE_solver(vfunc, DT_MICRO)

T_steps = TIME_SAMPLING * NUM_INTERVALS
ts = jnp.linspace(0.0, float(T_steps), T_steps)
_, Y_full = solver(ts=ts, y0=x0)               # [T_steps, B, 2, H, W]

Y = rearrange(Y_full, "T B C X Y -> B T C X Y")
Y_snap = Y[:, ::TIME_SAMPLING]                 # [B,K,2,H,W]
Y_snap = Y_snap[0]                             # [K,2,H,W]

# 3) Load trained NCA
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

# 4) Roll-out NCA (constant dt=1 per micro-step)
def rollout_nca(nca, x0, n_steps, n_intervals, key):
    traj = []
    x = x0
    for _ in range(n_intervals):
        traj.append(x)
        for _ in range(n_steps):
            key, subk = jr.split(key)
            x = nca(x, boundary_callback=lambda z: z, key=subk)
    return jnp.stack(traj, axis=0)  # [K, C, H, W]

hidden = jnp.zeros((CHANNELS-2, SIZE, SIZE), dtype=x0.dtype)
init_state = jnp.concatenate([Y_snap[0], hidden], axis=0)  # [C,H,W]

X_pred = rollout_nca(nca, init_state, TIME_SAMPLING, NUM_INTERVALS, jr.PRNGKey(42))
# X_pred: [K, C, H, W]

# 5) Build a tiny “validation” dataset just so the trainer hooks up its loss exactly
val_data = jnp.stack([
    jnp.concatenate([
        Y_snap,
        jnp.broadcast_to(hidden, (NUM_INTERVALS, *hidden.shape))
    ], axis=1)
], axis=0)  # [1, K, C, H, W]

trainer = NCA_Trainer(
    nca,
    val_data,
    model_filename=None,       # no saving/logging
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True,
    OBS_CHANNELS=2
)

# — patch in exactly the same per-step loss you used during training —
trainer._loss_func = loss.euclidean

# 6) Compute your per-interval validation loss
losses = []
for i in range(NUM_INTERVALS):
    x_i = X_pred[i]      # [C,H,W]
    y_i = Y_snap[i]      # [C,H,W]
    key_i = jr.fold_in(jr.PRNGKey(0), i)
    l = trainer.loss_func(
        x_i[None, ...],  # batch-dim = 1
        y_i[None, ...],
        key_i
    )
    losses.append(float(l[0]))

losses = jnp.array(losses)
mean_val_loss = float(jnp.mean(losses))

print("Validation loss per interval:", losses)
print(f"Overall mean validation loss = {mean_val_loss:.5e}")
