#!/usr/bin/env python
import jax
import jax.random as jr
import jax.numpy as jnp
import optax
import equinox as eqx
import os
import sys
import time

from einops import rearrange

# Add parent directory so we can import Common, PDE, and NCA modules
sys.path.append("..")

from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_keller_segel import F as F_keller_segel
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer as NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA

# --- Training hyperparameters ---
ITERS         = 4000        # total NCA‐training iterations
CHANNELS      = 16          # number of NCA hidden channels
SIZE          = 64          # spatial grid size (64×64)
BATCHES       = 6           # how many trajectories per batch
LEARN_RATE    = 1e-4        # base learning rate

# PDE “ground truth” parameters (must match the PDE script)
CELL_CHANNELS   = 1
SIGNAL_CHANNELS = 1
dx              = 0.5
dt_true         = 0.1       # Heun solver step size
SOLVER_PARAMS   = {
    "dt": dt_true,
    "SOLVER": "heun",
    "rtol": 1e-3,
    "atol": 1e-3,
    "ADAPTIVE": True,
    "DTYPE": "float32"
}

# Keller–Segel coefficients (matching the PDE training script)
alpha   = 0.01
c       = 3.8
D       = 0.8
epsilon = 0.1

# Trajectory parameters
T_FINAL          = 200.0
TIME_RESOLUTION  = 512   # number of saved frames between t=0 and t=200
TIME_SAMPLING    = 32    # we will downsample by 32, so final T = 256/32 = 8

# --- 1) Build random initial conditions (Uniform[0,0.1] for both channels) ---
def make_random_ic(key, B, H, W, CELL_CHANNELS, SIGNAL_CHANNELS):
    """
    Returns an array of shape [B, C_total, H, W], dtype=float32:
      - channel 0 (cell) ∼ Uniform(0, 0.1)
      - channel 1 (signal) = 0
    """
    key, subkey = jr.split(key)
    X = jr.uniform(
        subkey,
        shape=(B, CELL_CHANNELS + SIGNAL_CHANNELS, H, W),
        minval=0.0,
        maxval=0.1,
        dtype=jnp.float32
    )
    # Zero out the signal channel
    X = X.at[:, CELL_CHANNELS:].set(0.0)
    return X

key = jr.PRNGKey(0)
key, subkey = jr.split(key)

x0 = make_random_ic(
    subkey,
    B=BATCHES,
    H=SIZE,
    W=SIZE,
    CELL_CHANNELS=CELL_CHANNELS,
    SIGNAL_CHANNELS=SIGNAL_CHANNELS
)  # shape [B, 2, 64, 64], float32

# One pass of circular averaging to smooth out high-frequency noise
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=2)
v_av = eqx.filter_vmap(op.Average, in_axes=0, out_axes=0)
x0 = v_av(x0)  # shape still [B, 2, 64, 64]

# --- 2) Define RHS for Keller–Segel chemotaxis and build the “true” solver ---
func = F_keller_segel(
    PADDING="CIRCULAR",
    dx=dx,
    KERNEL_SCALE=1,
    alpha=alpha,
    c=c,
    D=D,
    epsilon=epsilon
)
vfunc = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vfunc, **SOLVER_PARAMS)

# --- 3) Integrate from t=0 to t=200, collect TIME_RESOLUTION frames ---
ts = jnp.linspace(0.0, T_FINAL, TIME_RESOLUTION, dtype=jnp.float32)
T_full, Y_full = solver(ts, x0)  
# Y_full has shape [TIME_RESOLUTION, BATCHES, 2, 64, 64]

# --- 4) Reshape to [BATCHES, TIME_RESOLUTION, 2, 64, 64] ---
Y_full = rearrange(Y_full, "T B C X Y -> B T C X Y")

# --- 5) Normalize both channels to [–1, +1] separately over the entire timeline ---
for ch in range(CELL_CHANNELS + SIGNAL_CHANNELS):
    ch_min = Y_full[:, :, ch].min()
    ch_max = Y_full[:, :, ch].max()
    Y_full = Y_full.at[:, :, ch].set((Y_full[:, :, ch] - ch_min) / (ch_max - ch_min))

# --- 6) Downsample in time by TIME_SAMPLING = 32 ---
Y = Y_full[:, ::TIME_SAMPLING, :, :, :]

# --- 7) Build NCA & Trainer (2-channel data) ---
nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "LAP", "GRAD"],
    ACTIVATION=jax.nn.relu,
    FIRE_RATE=1.0,
    key=key
)

trainer = NCA_Trainer(
    nca,
    Y,                                      # shape [BATCHES, 8, 2, 64, 64]
    model_filename="demo/train_nca_to_pde_keller_segel",
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

# --- 8) Set up optimizer ---
schedule  = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.995)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)

print("Saving to:", os.path.abspath("models/demo/train_nca_to_pde_keller_segel_unroll_trainer"))

# --- 9) Run training with TIME_SAMPLING = 32 ---
trainer.train(
    TIME_SAMPLING,  # 32 NCA‐updates per downsampled frame
    ITERS,          # total training iterations
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR="euclidean",
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=key
)
