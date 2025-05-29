#!/usr/bin/env python
import os, sys
import jax, jax.random as jr, jax.numpy as jnp
import equinox as eqx
from einops import rearrange

#  ─── adjust this path to point to your project root ───
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA

# 1) Hyperparameters (must match your training run)
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
BATCHES       = 1  # here we just do 1

# 2) Build a “ground-truth” Schnakenberg trajectory via PDE solver
key = jr.PRNGKey(0)
a, b, D = 0.2, 0.8, 50.0
U_eq = a + b
V_eq = b / (U_eq**2)
noise = 0.05

# initial condition [B,2,H,W]
key, k1 = jr.split(key)
U0 = U_eq + noise * jr.normal(k1, (BATCHES,1,SIZE,SIZE))
key, k2 = jr.split(key)
V0 = V_eq + noise * jr.normal(k2, (BATCHES,1,SIZE,SIZE))
x0 = jnp.concatenate([U0, V0], axis=1)

# smooth exactly as in training
op = Ops(PADDING=PADDING, dx=DX, KERNEL_SCALE=3)
for _ in range(3):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# build and run solver
func   = F_schnakenberg(PADDING=PADDING, dx=DX, KERNEL_SCALE=1,
                        a=a, b=b, D=D)
vfunc  = eqx.filter_vmap(func, in_axes=(None,0,None), out_axes=0)
solver = PDE_solver(vfunc, DT_MICRO)

T_steps = TIME_SAMPLING * NUM_INTERVALS
ts = jnp.linspace(0.0, float(T_steps), T_steps)
_, Y_full = solver(ts=ts, y0=x0)
# Y_full: [T_steps, B, 2, H, W]

# down-sample every TIME_SAMPLING steps
Y = rearrange(Y_full, "T B C X Y -> B T C X Y")
Y_snap = Y[:, ::TIME_SAMPLING]         # [B,NUM_INTERVALS,2,H,W]
Y_snap = Y_snap[0]                     # remove batch dim → [8,2,64,64]

# 3) Load your trained NCA
MODEL_FILE = "models/demo/train_nca_to_pde_schnakenberg.eqx"
dummy_nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE,
    KERNEL_SCALE=KERNEL_SCALE,
    key=jr.PRNGKey(0)
)
nca = eqx.tree_deserialise_leaves(MODEL_FILE, dummy_nca)

# 4) Roll‐out function (constant dt=1 per micro‐step)
def rollout_nca(nca, x0, n_steps, n_intervals, key):
    """
    x0: [C, H, W] initial state (must have C=CHANNELS)
    returns [n_intervals, C, H, W]
    """
    outs = []
    x = x0
    for _ in range(n_intervals):
        outs.append(x)
        for _ in range(n_steps):
            key, subk = jr.split(key)
            x = nca(x, boundary_callback=lambda z: z, key=subk)
    return jnp.stack(outs, axis=0)

# pad the 2 PDE channels up to CHANNELS with zeros
hidden = jnp.zeros((CHANNELS - 2, SIZE, SIZE), dtype=x0.dtype)
init_state = jnp.concatenate([Y_snap[0], hidden], axis=0)  # [8,64,64]

# run
X_pred = rollout_nca(nca, init_state, TIME_SAMPLING, NUM_INTERVALS, jr.PRNGKey(42))
# X_pred: [8, 8, 64, 64]

# 5) Compute MSE on U‐channel only
u_true = Y_snap[:, 0]     # [8,64,64]
u_pred = X_pred[:, 0]
mse_per_frame = jnp.mean((u_pred - u_true)**2, axis=(1,2))
mean_mse = float(jnp.mean(mse_per_frame))

print("MSE per interval:", mse_per_frame)
print(f"Overall mean  MSE = {mean_mse:.5e}")
