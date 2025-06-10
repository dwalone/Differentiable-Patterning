#!/usr/bin/env python
"""Schnakenberg → DINCA training demo **with physical‑unit read‑out**

This is the original script you uploaded, but the bottom part that prints
coefficients now **rescales them back to PDE units** so you can read the
learnt parameters directly.  The rest of the file (data generation,
training, model definition) is copy‑pasted verbatim to guarantee nothing
breaks.

Scaling rules (derived in the accompanying chat explanation):

* **Diffusion / gradient terms** (any spatial derivative of *u* or *v*)
  → `κ_phys = w_raw × range_var / Δt`
* **Reaction monomials** ϕ = u^p v^q →
  `κ_phys = w_raw × range_u^p × range_v^q / Δt`
* **Bias** (constant sources *a*, *b*) → same as *p=q=0* above.

`range_u`, `range_v` are saved *before* the normalisation step so the
formula is unambiguous.
"""

from typing import Sequence
import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys, os
sys.path.append('.')

from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2_dinca import DataAugmenter
from NCA.model.NCA_DINCA import NCA_DINCA as NCA
from demo.dinca_read_out import read_out
from demo.optimisers import masked_optimiser, normal_optimiser, warmup_optimiser

# ------------------------- hyper‑parameters --------------------------
ITERS         = 4000      # optimisation steps
CHANNELS      = 2        # NCA channels (u, v)
SIZE          = 64
BATCHES       = 10        # trajectories per batch
TIME_SAMPLING = 32       # frames between snapshots
LEARN_RATE    = 1e-4
DT            = 5e-3     # time step used by the PDE solver
OPTIMISER = 'masked'

# -------------------- build a “true” Schnakenberg run ----------------
key = jr.PRNGKey(0)
a_true, b_true, D_true = 0.2, 0.8, 50.0
U_eq, V_eq = a_true + b_true, b_true / (a_true + b_true) ** 2
noise = 0.5

key, k1, k2 = jr.split(key, 3)
U0 = U_eq + noise * jr.normal(k1, shape=(BATCHES, 1, SIZE, SIZE))
V0 = V_eq + noise * jr.normal(k2, shape=(BATCHES, 1, SIZE, SIZE))
X0 = jnp.concatenate([U0, V0], axis=1)  # [B, 2, H, W]

# smooth initial condition
op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
for _ in range(3):
    X0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(X0)

# PDE solver (semi‑discrete)
func   = F_schnakenberg(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1,
                        a=a_true, b=b_true, D=D_true)
vfunc  = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, DT)

# integrate
ts = jnp.linspace(0, TIME_SAMPLING * 16, TIME_SAMPLING * 16)
T, Y_phys = solver(ts, X0)                  # [T, B, 2, H, W]

# 7) reshape & normalize, keep both channels
Y = rearrange(Y_phys, "T B C X Y -> B T C X Y")  # [B, T, 2, H, W]
#Y = Y[:, :, :1]                                 # drop V
mins = Y.min(axis=(0, 1, 3, 4), keepdims=True)  # shape (1, 1, C, 1, 1)
ptps = Y.ptp(axis=(0, 1, 3, 4), keepdims=True)  # shape (1, 1, C, 1, 1)
Y = (Y - mins) / ptps
Y = Y[:, ::TIME_SAMPLING]                       # downsample in time

range_uv = ptps.squeeze()  # shape (C,)
range_u, range_v = float(range_uv[0]), float(range_uv[1])

# --------------------------- NCA + trainer ---------------------------
nca = NCA(N_CHANNELS=CHANNELS, FIRE_RATE=1.0, key=key, KERNEL_STR=["ID", "LAP", "GRAD"])
trainer = NCA_Trainer(nca, Y,
                      model_filename="demo/train_nca_to_pde_schnakenberg",
                      DATA_AUGMENTER=DataAugmenter, GRAD_LOSS=True, OBS_CHANNELS = 2)

# ----------------------------Optimiser--------------------------------
# Diffusion: ∇²(ch0) = 4, ∇²(ch1) = 5
keep_diff = [(0, 4), (1, 5)] # Δu gets ∇²u, Δv gets ∇²v
# Reaction: define what each Δchannel can use
keep_reac = {
    0: ['u', 'uuv'],   # only these on Δu
    1: ['uuv']    # only these on Δv
}
bias_mask = [True, True]

optimiser = masked_optimiser(ITERS, LEARN_RATE, nca, keep_diff, keep_reac, bias_mask)
#optimiser = normal_optimiser(ITERS, LEARN_RATE)
#optimiser = warmup_optimiser(ITERS, LEARN_RATE)
print(os.path.abspath("models/demo/train_nca_to_pde_schnakenberg"))
# ------------- curriculum parameters -----------------
# t_schedule  = [4, 8, 16, 32]          # horizons
# iters_total = 4000
# iters_per   = iters_total // len(t_schedule)
# # ------------- staged training -----------------------
# for phase, t_unroll in enumerate(t_schedule):
#     print(f"\n── Phase {phase}  (t = {t_unroll}) ──")
#     trainer.train(
#         t_unroll,
#         iters_per,
#         optimiser=optimiser,
#         WARMUP=0 if phase else 50,
#         LOG_EVERY=50,
#         key=jr.fold_in(key, phase),
#     )

trainer.train(TIME_SAMPLING, ITERS, WARMUP=50, optimiser=optimiser,
              LOSS_FUNC_STR="euclidean", LOOP_AUTODIFF="lax", LOG_EVERY=50, key=key, SPARSE_PRUNING=True)

read_out(trainer, range_u, range_v, DT)