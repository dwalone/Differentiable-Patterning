#!/usr/bin/env python
import jax, jax.random as jr, jax.numpy as jnp
import optax, equinox as eqx, sys, os
from einops import rearrange

sys.path.append('..')

from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_gierer_meinhardt import F as F_gm
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA

# ---------------- hyper-parameters (unchanged) ---------------------
ITERS, SIZE, TIME_SAMPLING, LEARN_RATE = 5000, 64, 16, 1e-4
CHANNELS, BATCHES = 8, 4         # NCA: 2 obs + 6 hidden

# ---------------- Gierer–Meinhardt constants -----------------------
D, a, b, c = 100.0, 0.5, 1.0, 6.1

# ---------------- initial condition (steady + noise) ---------------
key = jr.PRNGKey(0)
u0 = jnp.ones((BATCHES, 1, SIZE, SIZE))
v0 = jnp.ones((BATCHES, 1, SIZE, SIZE))
noise = 0.05
key, k1, k2 = jr.split(key, 3)
u0 += noise * jr.normal(k1, u0.shape)
v0 += noise * jr.normal(k2, v0.shape)
v0 = jnp.clip(v0, 1e-3, None)             # keep inhibitor positive
x0 = jnp.concatenate([u0, v0], axis=1)

op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
for _ in range(3):
    x0 = jax.vmap(op.Average, 0, 0)(x0)

# ---------------- RHS and solver -----------------------------------
func   = F_gm(PADDING="CIRCULAR", dx=1.0,
              a=a, b=b, c=c, D=D)
vfunc  = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, dt=1e-3)

ts = jnp.linspace(0, TIME_SAMPLING * 3, TIME_SAMPLING * 16)
T, Y = solver(ts, x0)                        # [T,B,2,H,W]
Y = rearrange(Y, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                              # keep u only
ptp = Y.max() - Y.min()
ptp = jnp.where(ptp < 1e-8, 1.0, ptp)      # avoid 0 divisor
Y   = (Y - Y.min()) / ptp                  # single, safe rescale
Y = Y[:, ::TIME_SAMPLING]

# ---------------- NCA and trainer ----------------------------------
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
    model_filename="demo/train_nca_to_pde_gierer_meinhardt",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

schedule  = optax.exponential_decay(LEARN_RATE, ITERS, 0.99)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)

trainer.train(
    TIME_SAMPLING,
    ITERS,
    optimiser=optimiser,
    WARMUP=50,
    LOSS_FUNC_STR="euclidean",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key
)
