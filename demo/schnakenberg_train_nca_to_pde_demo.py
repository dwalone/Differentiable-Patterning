#!/usr/bin/env python
import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys
import os
sys.path.append('..')

from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer_dynamic import NCA_Trainer_dynamic
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA

# training hyperparameters
ITERS         = 8000        # total training iterations
CHANNELS      = 8           # NCA channels
SIZE          = 64          # spatial grid size
BATCHES       = 2           # how many trajectories per batch
TIME_SAMPLING = 32          # frames between snapshots
T_max         = TIME_SAMPLING * 8    # e.g. 256
LEARN_RATE    = 1e-4        # base learning rate
weights = [1, 1, 2, 3, 5, 8, 13] # size(weights) + 1 number of snapshots, these weights represent intervals. change weights and length to change interval ratios and number of snapshots.

# build a “true” Schnakenberg trajectory
key       = jr.PRNGKey(int(time.time()))
a_true, b_true, D_true = 0.2, 0.8, 50.0
# steady state (a + b,  b/(a+b)^2)
U_eq      = a_true + b_true
V_eq      = b_true / (U_eq**2)
noise     = 0.05

def make_time_intervals(weights, total_time, batch_size):
    """
    Given unscaled weights of length K, returns:
      - dt_seq: [batch_size, K]  array of positive time‐steps summing to total_time
      - ts_ds:  [K+1]            array of cumulative times [0, t1, t2, …, total_time]
    """
    w = jnp.array(weights, dtype=jnp.float32)
    if jnp.any(w < 0):
        raise ValueError("Weights must be non‐negative")
    # normalize to sum=1
    w = w / jnp.sum(w)
    # scale to cover total_time
    dt = w * total_time          # length K, sums to total_time
    # cumulative times
    ts_ds = jnp.concatenate([jnp.array([0.], dtype=dt.dtype), jnp.cumsum(dt)])
    # broadcast to batch
    dt_seq = jnp.broadcast_to(dt[None, :], (batch_size, dt.shape[0]))
    return dt_seq, ts_ds

#Y = (Y - Y.min()) / (Y.max() - Y.min())         # normalize [0,1]
def normalize(batch):
    mn, mx = batch.min(), batch.max()
    return (batch - mn) / (mx - mn)

# initial condition ≈ steady‐state + noise
key, k1 = jr.split(key)
U0 = U_eq + noise * jr.normal(k1, shape=(BATCHES,1,SIZE,SIZE))
key, k2 = jr.split(key)
V0 = V_eq + noise * jr.normal(k2, shape=(BATCHES,1,SIZE,SIZE))
x0 = jnp.concatenate([U0, V0], axis=1)   # shape [B, 2, SIZE, SIZE]

# smooth out high-freq noise per batch
op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
for _ in range(3):
    # vmap the 3-D Average over the batch axis
    x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)

# define Schnakenberg RHS and solver
func   = F_schnakenberg(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1,
                        a=a_true, b=b_true, D=D_true)
# parallelise RHS over the batch axis
vfunc  = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, 5e-3)  # dt=5e-3

# 1) build dt_seq and ts_ds
dt_seq, ts_ds = make_time_intervals(weights, T_max, BATCHES)
# dt_seq.shape == (2,7), ts_ds.shape == (8,)

# 2) run solver on these exact snapshot times
T, Y_full = solver(ts=ts_ds, y0=x0)
# Y_full: [8, B, C, H, W]

# 3) reshape for training
Y = rearrange(Y_full, "T B C X Y -> B T C X Y")  # [B,8,C,H,W]
Y = jax.vmap(normalize)(Y)

#--- build NCA and trainer
nca = NCA(
    N_CHANNELS=CHANNELS,
    #KERNEL_STR=["ID", "LAP"],
    KERNEL_STR=["ID","LAP","GRAD"],
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=1.0,
    key=key
)

trainer = NCA_Trainer_dynamic(
    dt_seq,
    nca,
    Y,
    model_filename="demo/train_nca_to_pde_schnakenberg",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

#--- optimizer
schedule  = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)
print(os.path.abspath("models/demo/train_nca_to_pde_schnakenberg"))
#--- run training
trainer.train(
    TIME_SAMPLING,     # NCA steps per data snapshot
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR="euclidean",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key
)
