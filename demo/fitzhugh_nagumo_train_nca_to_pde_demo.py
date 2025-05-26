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

#--- Training hyperparameters
ITERS         = 8000        # total training iterations
CHANNELS      = 8           # NCA hidden channels
SIZE          = 64          # spatial grid size
BATCHES       = 1           # how many trajectories per batch
TIME_SAMPLING = 32          # solver steps between recorded frames
LEARN_RATE    = 1e-4        # base learning rate

#--- Build “true” FitzHugh–Nagumo trajectories
key = jr.PRNGKey(int(time.time()))

# FHN parameters (must match demo PDE)
D_true     = 0.1
eps_v_true = 0.01
a_v_true   = 0.5
a_z_true   = 0.1

# domain and discretization
dx = 1.0
L  = SIZE * dx

# initial condition: cosine mode m=4, v=0
m = 4
xs = jnp.linspace(0, L, SIZE, endpoint=False)
ys = xs
X, Y = jnp.meshgrid(xs, ys, indexing='ij')
u0 = jnp.cos(m * jnp.pi * X / L) * jnp.cos(m * jnp.pi * Y / L)
v0 = jnp.zeros_like(u0)
x0 = jnp.stack([u0, v0], axis=0)           # [2, H, W]
x0 = jnp.broadcast_to(x0, (BATCHES, 2, SIZE, SIZE))  # [B,2,H,W]

# smooth initial condition to avoid grid artifacts
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(2):
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# define RHS and solver
func  = F_fhn(
    PADDING="CIRCULAR",
    dx=dx,
    D=D_true,
    eps_v=eps_v_true,
    a_v=a_v_true,
    a_z=a_z_true
)
# parallelize over batch axis
vfunc = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
dt = 1e-2
solver = PDE_solver(vfunc, dt)

# integrate and collect snapshots
ts = jnp.linspace(0.0, TIME_SAMPLING * 8 * dt, TIME_SAMPLING * 8)
T, Y = solver(ts=ts, y0=x0)  # [T, B, 2, H, W]

# reshape and keep only u‐channel for training
Y = rearrange(Y, "T B C X Y -> B T C X Y")  # [B, T, 2, H, W]
# normalize each trajectory independently
def normalize(batch):
    mn, mx = batch.min(), batch.max()
    return (batch - mn) / (mx - mn)
Y = jax.vmap(normalize)(Y)

# downsample in time to frames every TIME_SAMPLING solver‐steps
Y = Y[:, ::TIME_SAMPLING]  # [B, 8, 2, H, W]

#--- Build NCA and trainer
nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "LAP", "GRAD"],
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=1.0,
    key=key
)

trainer = NCA_Trainer(
    nca,
    Y,
    model_filename="demo/train_nca_to_pde_fhn",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

#--- Optimizer
schedule  = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)

print("Saving to:", os.path.abspath("models/demo/train_nca_to_pde_fhn"))

#--- Run training
trainer.train(
    TIME_SAMPLING,    # NCA updates per data‐frame
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR="euclidean",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key
)
