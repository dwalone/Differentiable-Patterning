#!/usr/bin/env python
"""Gierer–Meinhardt → DINCA training demo (with physical read-out)."""

# ――― Imports (unchanged) ―――
from typing import Sequence
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx, sys, os, time
sys.path.append('.')

from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_gierer_meinhardt import F as F_gm   # ← CHANGED
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_DINCA import NCA_DINCA as NCA
from demo.dinca_read_out import read_out
from demo.optimisers import masked_optimiser

# -------------------- hyper-parameters -------------------------------
ITERS         = 8000
CHANNELS      = 2          # u, v
SIZE          = 64
BATCHES       = 1
TIME_SAMPLING = 16
LEARN_RATE    = 1e-5
DT            = 1e-3       # ← smaller dt for stiff D=100

# ---------------------- ground-truth PDE run -------------------------
key = jr.PRNGKey(0)
D_true, a_true, b_true, c_true = 100.0, 0.5, 1.0, 6.1

# initial condition: one dot           (change as you like)
u0 = jnp.ones((BATCHES, 1, SIZE, SIZE))
v0 = jnp.ones((BATCHES, 1, SIZE, SIZE))
yy, xx = jnp.ogrid[:SIZE, :SIZE]
gauss = jnp.exp(-((xx-SIZE//2)**2 + (yy-SIZE//2)**2) / 4.0)
u0 += 0.30 * gauss[None, None]
v0 = jnp.clip(v0, 1e-3, None)           # keep v positive

X0 = jnp.concatenate([u0, v0], axis=1)

# optional smoothing
op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
for _ in range(3):
    X0 = jax.vmap(op.Average, 0, 0)(X0)

func   = F_gm(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1,
              a=a_true, b=b_true, c=c_true, D=D_true)
vfunc  = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, DT)

ts = jnp.linspace(0, TIME_SAMPLING * 5, TIME_SAMPLING * 16)
T, Y_phys = solver(ts, X0)                       # [T,B,2,H,W]

Y = rearrange(Y_phys, "T B C X Y -> B T C X Y")
mins = Y.min(axis=(0, 1, 3, 4), keepdims=True)
ptps = jnp.maximum(Y.ptp(axis=(0, 1, 3, 4), keepdims=True), 1e-8)
Y = (Y - mins) / ptps
Y = Y[:, ::TIME_SAMPLING]                       # [B,16,2,H,W]

range_u, range_v = map(float, ptps.squeeze())

# --------------------------- DINCA model -----------------------------
nca = NCA(N_CHANNELS=CHANNELS, FIRE_RATE=0.5,
          KERNEL_STR=["ID", "LAP", "GRAD"], key=key)

trainer = NCA_Trainer(
    nca,
    Y,
    model_filename="demo/train_nca_to_pde_gierer_meinhardt",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True,
    OBS_CHANNELS=2,
)

# ------------------------ masked optimiser ---------------------------
# Diffusion indices remain identical: Δu ← ∇²u (4), Δv ← ∇²v (5)
keep_diff = [(0, 4), (1, 5)]

# Reaction monomials needed:
#   Δu: 1, u, v, uu
#   Δv: uu, v
keep_reac = {
    0: ['1', 'u', 'v', 'uu'],
    1: ['uu', 'v'],
}

bias_mask = [False, False]    # we rely on constant feature '1'

optimiser = masked_optimiser(
    ITERS, LEARN_RATE, nca, keep_diff, keep_reac, bias_mask
)

# ----------------------------- train ---------------------------------
trainer.train(
    TIME_SAMPLING,
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR="l2",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key,
    SPARSE_PRUNING=False,
)

read_out(trainer, range_u, range_v, DT)
