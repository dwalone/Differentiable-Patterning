#!/usr/bin/env python
import os
import time

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from einops import rearrange

import sys
sys.path.append('..')

from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_fitzhugh_nagumo import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA

#--- Training hyperparameters (unchanged)
ITERS         = 20000
CHANNELS      = 16
SIZE          = 64
BATCHES       = 2
TIME_SAMPLING = 50
LEARN_RATE    = 5e-5

# ----------------------------------------------------------------------
# 1) make_spike_ic in float32
# ----------------------------------------------------------------------
def make_spike_ic(key, B, H, W, N_clicks=5, sigma=1.5, amplitude=1.0):
    """
    Returns [B,2,H,W] in float32 with N_clicks Gaussian bumps in channel 0.
    Channel 1 is zero.
    """
    # 1) precompute Gaussian kernel in float32
    radius = int(3 * sigma)
    xs = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    ys = xs
    Xg, Yg = jnp.meshgrid(xs, ys, indexing='ij')
    kernel = amplitude * jnp.exp(-(Xg**2 + Yg**2) / (2 * sigma**2))

    # 2) init U,V in float32
    U0 = jnp.zeros((B, H, W), dtype=jnp.float32)
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
    return jnp.stack([U0, V0], axis=1)  # shape [B,2,H,W] dtype=float32

# ----------------------------------------------------------------------
# 2) Build “true” FitzHugh–Nagumo data (float32)
# ----------------------------------------------------------------------
key = jr.PRNGKey(int(time.time()))

# FHN parameters
D_true     = 0.05
eps_v_true = 0.005
a_v_true   = 0.5
a_z_true   = 0.0

# domain and discretization
dx = 1.0
dt = jnp.float32(1e-2)

# 3) IC in float32
key, subkey = jr.split(key)
x0 = make_spike_ic(
    subkey,
    B=BATCHES,
    H=SIZE,
    W=SIZE,
    N_clicks=8,
    sigma=1.0,
    amplitude=1.0
)   # [B,2,H,W] float32

# 4) smooth in float32
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(2):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# ----------------------------------------------------------------------
# 5) RHS and solver (float32)
# ----------------------------------------------------------------------
func  = F_fhn(
    PADDING="CIRCULAR",
    dx=dx,
    D=D_true,
    eps_v=eps_v_true,
    a_v=a_v_true,
    a_z=a_z_true
)
vfunc = eqx.filter_vmap(func, in_axes=(None,0,None), out_axes=0)
solver = PDE_solver(vfunc, dt)

# 6) Integrate & collect snapshots in float32
#    here ts is float32, so the solver sees float32 everywhere
ts = jnp.linspace(
    0.0, TIME_SAMPLING * 8 * float(dt),
    TIME_SAMPLING * 8,
    dtype=jnp.float32
)
T, Y = solver(ts=ts, y0=x0)  # Y dtype=float32

# 7) reshape & normalize
Y = rearrange(Y, "T B C X Y -> B T C X Y")  # [B,T,2,H,W] float32
def normalize(batch):
    mn, mx = batch.min(), batch.max()
    return (batch - mn) / (mx - mn)
Y = jax.vmap(normalize)(Y)

# 8) downsample in time
Y = Y[:, ::TIME_SAMPLING]  # [B,8,2,H,W] float32

# ----------------------------------------------------------------------
# 9) Build NCA & trainer
# ----------------------------------------------------------------------
nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID","LAP","GRAD"],
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=1.0,
    key=key
)

trainer = NCA_Trainer(
    nca,
    Y,
    model_filename="demo/train_nca_to_pde_fhn_click",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

schedule  = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))

print("Saving to:", os.path.abspath("models/demo/train_nca_to_pde_fhn_click"))

# 10) run training
trainer.train(
    TIME_SAMPLING,
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR="euclidean",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key
)