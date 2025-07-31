#!/usr/bin/env python
import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys
import os
sys.path.append('../..')

print("1")

from einops import rearrange
print("2")
from Common.model.spatial_operators import Ops
print("3")
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
print("4")
from PDE.model.solver.semidiscrete_solver import PDE_solver
print("5")
from NCA.trainer.NCA_trainer import NCA_Trainer
print("6")
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
print("7")
from NCA.model.NCA_model import NCA

print("a")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--time_sampling", type=int,   default=32)
parser.add_argument("--learn_rate",    type=float, default=5e-4)
parser.add_argument("--channels",      type=int,   default=8)
parser.add_argument("--loss",          type=str,   default="euclidean")
parser.add_argument("--model_filename",default="demo/train_nca_to_pde_schnakenberg")
parser.add_argument("--fire_rate",    type=float, default=1.0)
parser.add_argument("--state_reg",    type=float, default=1.0)
args = parser.parse_args()

print("b")

# training hyperparameters
ITERS         = 8000        # total training iterations
CHANNELS      = args.channels           # NCA channels
SIZE          = 64          # spatial grid size
#BATCHES       = 8           # how many trajectories per batch
TIME_SAMPLING = args.time_sampling          # frames between snapshots
LEARN_RATE    = args.learn_rate        # base learning rate
LOSS_FUNC_STR = args.loss
MODEL_DIR     = args.model_filename
FIRE_RATE     = args.fire_rate
STATE_REGULARISER = args.state_reg

# build a “true” Schnakenberg trajectory
key       = jr.PRNGKey(0)
a_true, b_true, D_true = 0.01, 2, 80

print("c")

#grid
#######################################
# # steady state (a + b,  b/(a+b)^2)
# U_eq      = a_true + b_true
# V_eq      = b_true / (U_eq**2)
# noise     = 0.05

# # initial condition ≈ steady‐state + noise
# key, k1 = jr.split(key)
# U0 = U_eq + noise * jr.normal(k1, shape=(BATCHES,1,SIZE,SIZE))
# key, k2 = jr.split(key)
# V0 = V_eq + noise * jr.normal(k2, shape=(BATCHES,1,SIZE,SIZE))
# x0 = jnp.concatenate([U0, V0], axis=1)   # shape [B, 2, SIZE, SIZE]

# # smooth out high-freq noise per batch
# op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
# for _ in range(3):
#     # vmap the 3-D Average over the batch axis
#     x0 = jax.vmap(op.Average, in_axes=0, out_axes=0)(x0)
##########################################

# ####################################
# # dot
# # steady state (a + b,  b/(a+b)^2)
# U_eq = a_true + b_true
# V_eq = b_true / (U_eq**2)

# # ------------ new single-dot initial condition -----------------------
# # homogeneous steady state everywhere …
# U0 = jnp.full((BATCHES, 1, SIZE, SIZE), U_eq)
# V0 = jnp.full((BATCHES, 1, SIZE, SIZE), V_eq)
# # … except one activator peak at the lattice centre
# cx, cy = SIZE // 2, SIZE // 2
# U0 = U0.at[:, 0, cx, cy].set(U_eq + 0.2)   # 0.2 ≈ 10 % bump, adjust as needed
# x0 = jnp.concatenate([U0, V0], axis=1)     # shape [B, 2, SIZE, SIZE]
# ######################################

# ##########################################
# U_eq = a_true + b_true
# V_eq = b_true / (U_eq ** 2)

# # start from homogeneous steady state
# U0 = jnp.full((BATCHES, 1, SIZE, SIZE), U_eq)
# V0 = jnp.full((BATCHES, 1, SIZE, SIZE), V_eq)

# # sprinkle exactly two peaks at random positions for each batch
# key, sub = jr.split(key)
# coords = jr.randint(sub, (BATCHES, 2, 2), 0, SIZE)   # shape (B, 2 dots, 2 coords)

# def add_two_peaks(u_slice, xy):
#     """u_slice : (1,H,W), xy : (2,2)"""
#     for k in range(2):
#         x, y = xy[k]
#         u_slice = u_slice.at[0, x, y].set(U_eq + 0.2)
#     return u_slice

# U0 = jax.vmap(add_two_peaks, in_axes=(0, 0))(U0, coords)

# x0 = jnp.concatenate([U0, V0], axis=1)    # (B, 2, SIZE, SIZE)
# ###################

# ----------------------------------------------------------------------
# NEW: homogeneous steady state  +  small Gaussian noise
# ----------------------------------------------------------------------
# steady-state values            (u*, v*) = (a + b,  b / (a + b)^2)
# U_eq = a_true + b_true
# V_eq = b_true / (U_eq**2)

# sigma = 0.03          # noise amplitude  (3 % of steady state)
# key, sub1, sub2 = jr.split(key, 3)

# # sample independent noise for each batch item
# U0 = U_eq + sigma * jr.normal(sub1, shape=(BATCHES, 1, SIZE, SIZE))
# V0 = V_eq + sigma * jr.normal(sub2, shape=(BATCHES, 1, SIZE, SIZE))

# x0 = jnp.concatenate([U0, V0], axis=1)   # shape (B, 2, SIZE, SIZE)
# ----------------------------------------------------------------------
print("d")
#######################################################################
U_eq = a_true + b_true
V_eq = b_true / (U_eq**2)
sigma = 0.03
import jax.lax as lax

def make_ic(key, choice: jnp.ndarray):
    """
    key     : PRNGKey
    choice  : scalar int32  (0=noise, 1=central dot, 2=two dots, 3=grid)
    returns : (2, SIZE, SIZE) tensor
    """
    k1, k2 = jr.split(key)          # k1 for noise/dots, k2 for coords
    # ------ helper: homogeneous steady state --------------------------
    def _base():
        U = jnp.full((SIZE, SIZE), U_eq)
        V = jnp.full_like(U, V_eq)
        return U, V

    # -------- four deterministic branches -----------------------------
    def branch_noise(_):
        U, V = _base()
        rng1, rng2 = jr.split(k1)
        U += sigma * jr.normal(rng1, (SIZE, SIZE))
        V += sigma * jr.normal(rng2, (SIZE, SIZE))
        return jnp.stack([U, V])

    def branch_central(_):
        U, V = _base()
        U = U.at[SIZE//2, SIZE//2].set(U_eq + 0.2)
        return jnp.stack([U, V])

    def branch_two(_):
        U, V = _base()
        xy = jr.randint(k1, (2, 2), 0, SIZE)
        U = U.at[xy[:, 0], xy[:, 1]].add(0.2)
        return jnp.stack([U, V])

    def _scatter_n_dots(U, rng, n):
        xy = jr.randint(rng, (n, 2), 0, SIZE)   # (n,2)
        return U.at[xy[:, 0], xy[:, 1]].add(0.2)

    # ---------- branch 3 : three random dots ----------------------------
    def branch_three(k):
        U, V = _base()
        U = _scatter_n_dots(U, k, 3)
        return jnp.stack([U, V])

    # ---------- branch 4 : four random dots -----------------------------
    def branch_four(k):
        U, V = _base()
        U = _scatter_n_dots(U, k, 4)
        return jnp.stack([U, V])

    def branch_grid(_):
        U, V = _base()
        step = SIZE // 4
        ix   = jnp.arange(0, SIZE, step)
        coords = jnp.stack(jnp.meshgrid(ix, ix, indexing="ij"), -1).reshape(-1, 2)
        U = U.at[coords[:, 0], coords[:, 1]].set(U_eq + 0.15)
        return jnp.stack([U, V])

    return lax.switch(
        choice,
        (
            branch_noise,    # 0
            branch_central,  # 1
            branch_two,      # 2
            branch_three,    # 3
            branch_four,     # 4
            branch_grid      # 5
        ),
        k2                      # <- PRNGKey forwarded to the chosen branch
    )

print("e")

mix = {
        0: 2,
        1: 1,    # one central dot
        2: 1,
        3: 1,
        4: 1,
       }
BATCHES = sum(mix.values())
print("f")

# master key and a key per batch element
key, *subkeys = jr.split(key, BATCHES + 1)
subkeys = jnp.array(subkeys)                 # shape (BATCHES, 2)

# explicit choice list  e.g. [1, 2, 2, 0, 0, 0, 3, 3]
choices = jnp.concatenate([
    jnp.full(n, c, dtype=jnp.int32) for c, n in mix.items()
])
print("g")
# vectorised call over (key, choice)
x0 = jax.vmap(make_ic)(subkeys, choices)     # (BATCHES, 2, SIZE, SIZE)

##########################################################################

# define Schnakenberg RHS and solver
func   = F_schnakenberg(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=2,
                        a=a_true, b=b_true, D=D_true)
# parallelise RHS over the batch axis
vfunc  = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, 5e-3)  # dt=5e-3

# integrate and collect snapshots
sampling_constant = 32
ts = jnp.linspace(0, sampling_constant * 3, sampling_constant * 10)
T, Y = solver(ts, x0)
# Y has shape [T, B, 2, SIZE, SIZE]

# reshape and keep only U-channel for NCA training
Y = rearrange(Y, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                                 # drop V
Y = (Y - Y.min()) / (Y.max() - Y.min())         # normalize [0,1]
Y = Y[:, ::sampling_constant]                       # downsample in time
print("h")
#--- build NCA and trainer
nca = NCA(
    N_CHANNELS=CHANNELS,
    #KERNEL_STR=["ID", "LAP"],
    KERNEL_STR=["ID","LAP","GRAD"],
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=FIRE_RATE,
    key=key
)
print("i")
trainer = NCA_Trainer(
    nca,
    Y,
    model_filename=MODEL_DIR,
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)
print("j")
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
    LOSS_FUNC_STR=LOSS_FUNC_STR,
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key,
    STATE_REGULARISER=STATE_REGULARISER
)
