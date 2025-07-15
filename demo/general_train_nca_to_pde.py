#!/usr/bin/env python
# general_train_nca_to_pde.py
# ================================================================
import time, sys, os, argparse
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.fixed_models.update_gray_scott_new import F as F_gray_scott
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott_old
from PDE.model.fixed_models.update_fhn import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA
import jax.lax as lax
import numpy as np

# ------------------------------------------------------------------
# 0 · Command-line interface
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pde",            default="schnakenberg",
                    choices=["schnakenberg", "fhn", "gs_glider", "gs_labyrinth", "gs_uskate", "gs_old"])
parser.add_argument("--time_sampling",  type=int,   default=32)
parser.add_argument("--learn_rate",     type=float, default=5e-4)
parser.add_argument("--channels",       type=int,   default=16)
parser.add_argument("--loss",           default="euclidean")
parser.add_argument("--model_filename", default="demo/train_nca_to_pde")
parser.add_argument("--fire_rate",      type=float, default=1.0)
parser.add_argument("--state_reg",      type=float, default=1.0)
args = parser.parse_args()

# ------------------------------------------------------------------
# 1 · Global training constants
# ------------------------------------------------------------------
ITERS         = 8_000
SIZE          = 64
TIME_SAMPLING = args.time_sampling
CHANNELS      = args.channels
LEARN_RATE    = args.learn_rate
LOSS_FUNC_STR = args.loss
FIRE_RATE     = args.fire_rate
STATE_REGULARISER = args.state_reg
MODEL_DIR     = f"{args.model_filename}_{args.pde}"
RADIUS = 6   # 3 → 7×7 square


# ------------------------------------------------------------------
# 2 · PDE-specific configuration
# ------------------------------------------------------------------
PDE_CONFIGS = {  # all floats (jnp)
    "schnakenberg": dict(a=0.01,  b=2.0,   D=80.0,
                         steady=lambda p: (p["a"]+p["b"],
                                            p["b"]/(p["a"]+p["b"])**2)),
    "fhn":          dict(D=20.0, eps_v=0.5, a_v=1.0, a_z=-0.1,
                         steady=lambda p: (0.0, 0.0)),
    "gs_glider":    dict(a=0.014, b=0.054, D=2.0,
                         steady=lambda p: (0.0, 1.0)),  # U=1,V=0 in GS conv.
    "gs_labyrinth": dict(a=0.037, b=0.06, D=2.0,
                         steady=lambda p: (0.0, 1.0)),
    "gs_uskate":    dict(a=0.062, b=0.061, D=2.0,
                         steady=lambda p: (0.0, 1.0)),
    "gs_old":       dict(DA=0.1,DB=0.05,alpha=0.06230,gamma=0.06268,
                         steady=lambda p: (0.0, 0.0)),
}

cfg = PDE_CONFIGS[args.pde]

# factory for RHS
def make_rhs(pde_name, **pars):
    if pde_name == "schnakenberg":
        return F_schnakenberg(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1, **pars)
    if pde_name == "fhn":
        return F_fhn(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1, **pars)
    if pde_name == "gs_old":
        return F_gray_scott_old(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1, **pars)
    if pde_name in ["gs_glider", "gs_labyrinth", "gs_uskate"]:
        return F_gray_scott(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=1, **pars)
    raise ValueError("unknown PDE")

rhs   = make_rhs(args.pde, **{k: v for k, v in cfg.items() if k != "steady"})
v_rhs = eqx.filter_vmap(rhs, in_axes=(None, 0, None), out_axes=0)
if args.pde.startswith("gs"):
    dt = 0.2
else:
    dt = 5e-3
solver= PDE_solver(v_rhs, dt=dt)

# ------------------------------------------------------------------
# 3 · Initial-condition generator (unchanged except steady state)
# ------------------------------------------------------------------
U_eq, V_eq = cfg["steady"](cfg)
sigma      = 0.03

def make_ic(key, choice: jnp.ndarray):
    k1, k2 = jr.split(key)

    def _base():
        U = jnp.full((SIZE, SIZE), U_eq)
        V = jnp.full_like(U, V_eq)
        return U, V

    def noise(_):
        U, V = _base()
        r1, r2 = jr.split(k1)
        U += sigma * jr.normal(r1, (SIZE, SIZE))
        V += sigma * jr.normal(r2, (SIZE, SIZE))
        return jnp.stack([U, V])

    def central(_):
        U, V = _base()
        U = _scatter(U, k2, n=1, delta=0.2, radius=RADIUS)   # use same helper
        return jnp.stack([U, V])


    def gaussian_patch(r, delta=0.2, sigma=1.5):
        ax = jnp.arange(-r, r+1)
        g  = jnp.exp(-(ax**2)/(2*sigma**2))
        kern = jnp.outer(g, g)
        return kern / kern.max() * delta

    # ------------------------------------------------------------------
    # Gray–Scott “inverted” circular blobs
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Gray–Scott OLD : inverted blobs (no square artefacts, JIT-safe) ---
    def gs_inverted_blob(rng, n=1, radius=SIZE // 8):
        """
        Return U,V  (H×H) with white background (U=1,V=0) and
        n circular blobs where U=0, V=1.  Works inside JIT / vmap.
        """
        H   = SIZE
        U   = jnp.ones((H, H), dtype=jnp.float32)   # background
        V   = jnp.zeros_like(U)

        coords  = jnp.arange(-radius, radius + 1)
        xx, yy  = jnp.meshgrid(coords, coords, indexing="ij")
        circle  = ((xx**2 + yy**2) <= radius**2).astype(U.dtype)   # 1 inside
        patchU  = 1.0 - circle     # 0 inside, 1 outside (matches bg)
        patchV  =        circle    # 1 inside, 0 outside

        xy = jr.randint(rng, (n, 2), radius, H - radius)           # centres

        def body(carry, centre):
            A, B = carry
            x, y = centre
            idx  = (x - radius, y - radius)        # top-left corner
            A    = lax.dynamic_update_slice(A, patchU, idx)
            B    = lax.dynamic_update_slice(B, patchV, idx)
            return (A, B), None

        (U, V), _ = lax.scan(body, (U, V), xy)
        return U, V


    def _scatter(U, rng, n, delta=0.2, radius=3):
        """
        Add `n` square patches of side (2*radius+1) to U.
        delta  : peak intensity to add
        radius : 0 → 1-pixel, 1 → 3×3, 2 → 5×5, …

        Returns updated U (V untouched).
        """
        patch_size = 2 * radius + 1
        patch = gaussian_patch(radius, delta)

        # choose centres at least `radius` pixels from the border
        xy = jr.randint(rng, (n, 2), radius, SIZE - radius)

        def body(carry, coords):
            A = carry
            x, y = coords
            A = lax.dynamic_update_slice(A, patch, (x - radius, y - radius))
            return A, None

        U, _ = lax.scan(body, U, xy)
        return U

    def two(_):
        if args.pde == "gs_old":
            U, V = gs_inverted_blob(k2, n=2)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            U = _scatter(U, k2, n=2, delta=0.2, radius=RADIUS)
            return jnp.stack([U, V])
    def three(k): 
        if args.pde == "gs_old":
            U, V = gs_inverted_blob(k2, n=3)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            return jnp.stack([_scatter(U, k, n=3, delta=0.2, radius=RADIUS), V])
    def four(k):  
        if args.pde == "gs_old":
            U, V = gs_inverted_blob(k2, n=4)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            return jnp.stack([_scatter(U, k, n=4, delta=0.2, radius=RADIUS), V])

    return lax.switch(choice,
                      (noise, central, two, three, four),
                      k2)

mix = {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}
BATCHES = sum(mix.values())
key, *sub = jr.split(jr.PRNGKey(0), BATCHES + 1)
sub = jnp.array(sub)
choices = jnp.concatenate([jnp.full(n, c, jnp.int32)
                           for c, n in mix.items()])
x0 = jax.vmap(make_ic)(sub, choices)         # (B,2,H,W)

# ------------------------------------------------------------------
# 4 · Ground-truth trajectories
# ------------------------------------------------------------------
sampling_constant = 32
if args.pde.startswith("gs"):
    ts = jnp.linspace(0, 10000, sampling_constant * 10)
else:
    ts = jnp.linspace(0, sampling_constant * 3, sampling_constant * 10)
T, Y = solver(ts, x0)
# reshape and keep only U-channel for NCA training
Y = rearrange(Y, "T B C X Y -> B T C X Y")
Y = Y[:, :, :1]                             # keep only U for others
Y = (Y - Y.min()) / (Y.max() - Y.min())     # normalise after slice
Y = Y[:, ::sampling_constant]                   # downsample in time

# ------------------------------------------------------------------
# 5 · Build NCA + trainer
# ------------------------------------------------------------------
nca = NCA(N_CHANNELS=CHANNELS,
          KERNEL_STR=["ID", "LAP", "GRAD"],
          ACTIVATION=jax.nn.relu,
          FIRE_RATE=FIRE_RATE,
          key=jr.PRNGKey(1))

trainer = NCA_Trainer(
    nca, Y,
    model_filename=MODEL_DIR,
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

schedule = optax.exponential_decay(
    LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)

trainer.train(
    TIME_SAMPLING,
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR=LOSS_FUNC_STR,
    LOOP_AUTODIFF="lax",
    LOG_EVERY=50,
    key=jr.PRNGKey(2),
    STATE_REGULARISER=STATE_REGULARISER
)
