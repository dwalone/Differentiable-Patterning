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
from NCA.trainer.data_augmenter_nca_from_pde_2_chemotaxis import DataAugmenter
from NCA.model.NCA_DINCA import NCA_DINCA as NCA

# ------------------------- hyper‑parameters --------------------------
ITERS         = 1000      # optimisation steps
CHANNELS      = 2        # NCA channels (u, v)
SIZE          = 64
BATCHES       = 4        # trajectories per batch
TIME_SAMPLING = 32       # frames between snapshots
LEARN_RATE    = 1e-4
DT            = 5e-3     # time step used by the PDE solver

# -------------------- build a “true” Schnakenberg run ----------------
key = jr.PRNGKey(0)
a_true, b_true, D_true = 0.2, 0.8, 50.0
U_eq, V_eq = a_true + b_true, b_true / (a_true + b_true) ** 2
noise = 0.05

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
ts = jnp.linspace(0, TIME_SAMPLING * 8, TIME_SAMPLING * 8)
T, Y_phys = solver(ts, X0)                  # [T, B, 2, H, W]

# ------------------------------------------------ save ranges *before* normalisation
range_uv = jnp.ptp(Y_phys, axis=(0, 1, 3, 4))  # shape (2,)
range_u, range_v = float(range_uv[0]), float(range_uv[1])

# -------------- prepare training data  (keep only U‑channel) ---------
Y = rearrange(Y_phys, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                                         # drop V
Y = (Y - Y.min()) / (Y.max() - Y.min())                 # [0,1]
Y = Y[:, ::TIME_SAMPLING]                               # temporal stride

# --------------------------- NCA + trainer ---------------------------
nca = NCA(N_CHANNELS=CHANNELS, FIRE_RATE=1.0, L1_COEFF=1e-3, key=key)
trainer = NCA_Trainer(nca, Y,
                      model_filename="demo/train_nca_to_pde_schnakenberg",
                      DATA_AUGMENTER=DataAugmenter, GRAD_LOSS=True)

schedule  = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))

print(os.path.abspath("models/demo/train_nca_to_pde_schnakenberg"))
trainer.train(TIME_SAMPLING, ITERS, WARMUP=50, optimiser=optimiser,
              LOSS_FUNC_STR="euclidean", LOOP_AUTODIFF="lax", LOG_EVERY=50, key=key)

# ============================ read‑out ===============================
_, w_raw, b_raw = trainer.NCA_model.get_weights()
w_raw = jnp.squeeze(w_raw)  # (C_out, F)
C = trainer.NCA_model.N_CHANNELS
K_diff = 3 * C
w_diff_raw = w_raw[:, :K_diff]
w_reac_raw = w_raw[:, K_diff:]

# ---------- helper: label → exponents --------------------------------
reac_labels = trainer.NCA_model.reaction_labels

def label_to_exponents(lbl: str):
    lbl = lbl.strip()
    return lbl.count('u'), lbl.count('v')  # power of u, power of v

# ---------- rescale to physical units ---------------------------------
scale_uv = jnp.array([range_u, range_v])

# diffusion / gradient (same formula for all spatial derivatives)
w_diff_phys = jnp.zeros_like(w_diff_raw)
for j, label in enumerate([f"∂x", "∂y", "∇²"]):
    for var in range(C):
        idx = 3 * var + j  # because of sorting order in script
        w_diff_phys = w_diff_phys.at[:, idx].set(
            w_diff_raw[:, idx] * scale_uv[var] / DT
        )

# reaction monomials
w_reac_phys = jnp.zeros_like(w_reac_raw)
for k, lbl in enumerate(reac_labels):
    pu, pv = label_to_exponents(lbl)
    scale = (scale_uv[0] ** pu) * (scale_uv[1] ** pv) / DT
    w_reac_phys = w_reac_phys.at[:, k].set(w_reac_raw[:, k] * scale)

# biases (constant sources)
bias_phys = b_raw * scale_uv / DT  # channel‑wise

# ---------- pretty print ----------------------------------------------
diff_labels = [f"{d}(ch{c})" for d in ("∂x", "∂y", "∇²") for c in range(C)]
print("\n=== DIFFUSION FEATURES (physical units) ===")
for c_out in range(C):
    print(f"\nΔChannel {c_out}:")
    for i, label in enumerate(diff_labels):
        print(f"  {label:10s}: {float(w_diff_phys[c_out, i]):+.5e}")

print("\n=== REACTION TERMS (physical units) ===")
for c_out in range(C):
    print(f"\nΔChannel {c_out}:")
    for i, lbl in enumerate(reac_labels):
        print(f"  {lbl:10s}: {float(w_reac_phys[c_out, i]):+.5e}")

print("\n=== CONSTANT SOURCES (bias) ===")
print(f"  a (u‑eqn): {float(bias_phys[0]):+.5e}")
print(f"  b (v‑eqn): {float(bias_phys[1]):+.5e}")
