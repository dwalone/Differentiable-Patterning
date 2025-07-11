#!/usr/bin/env python
# schnakenberg_train_npde_demo.py
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx, time, sys, os
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F          # neural Schnakenberg RHS
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer                   # << new
# ------------------------------------------------------------------
# hyper-params
ITERS, SIZE, BATCHES, DT0 = 4000, 64, 1, 5e-3
TIME_SAMPLING, LEARN_RATE  = 32, 1e-4
CHANNELS = 2
# ------------------------------------------------------------------
# build ground-truth Schnakenberg movie (unchanged)
key     = jr.PRNGKey(0)
a, b, D = 0.2, 0.8, 50.0
Ueq, Veq = a+b, b/(a+b)**2
noise   = 0.05
key, k1, k2 = jr.split(key,3)
U0 = Ueq + noise*jr.normal(k1,(BATCHES,1,SIZE,SIZE))
V0 = Veq + noise*jr.normal(k2,(BATCHES,1,SIZE,SIZE))
x0 = jnp.concatenate([U0,V0],axis=1)

# smooth initial condition (unchanged)
op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
for _ in range(3):
    x0 = jax.vmap(op.Average,in_axes=0,out_axes=0)(x0)

# -------- build **learnable** nPDE  --------------------------------
func      = F(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1,
              a=a, b=b, D=D, N_CHANNELS = CHANNELS)                   # ↞ same class but **trainable**
vfunc     = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
solver    = PDE_solver(vfunc, DT0)             # Diffrax wrapper

# -------- prepare training data & time grid ------------------------
ts = jnp.linspace(0, TIME_SAMPLING*8, TIME_SAMPLING*8)
_, Y = solver(ts, x0)                          # generate targets
Y  = rearrange(Y,"T B C X Y -> B T C X Y")     # [B,T,C,H,W]
Ts = jnp.broadcast_to(ts,(BATCHES,ts.size))    # [B,T]

# keep every k-th frame so the model learns to **integrate** between them
Y  = Y[:,::TIME_SAMPLING]                      # shape [B,8,C,H,W]
Ts = Ts[:,::TIME_SAMPLING]                     # matching time stamps

# -------- instantiate trainer --------------------------------------
pde_hparams = dict(PADDING="CIRCULAR",dx=1.0)  # for checkpoints
trainer = PDE_Trainer(solver, pde_hparams,
                      data=Y, Ts=Ts,
                      model_filename="demo/train_npde_schnak")

# -------- optimiser: AdamW + global-norm clip ----------------------
optimiser = optax.chain(
    optax.clip_by_global_norm(1.0),            # << new safety net
    optax.adamw(LEARN_RATE)
)

# -------- run training ---------------------------------------------
trainer.train(SUBTRAJECTORY_LENGTH=1,          # fit single-step gaps
              TRAINING_ITERATIONS=ITERS,
              OPTIMISER=optimiser,
              WARMUP=50,
              LOG_EVERY=50)
