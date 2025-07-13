#!/usr/bin/env python
# schnakenberg_nca_batch_inference.py
# ------------------------------------------------------------
import argparse, os, re, sys, json
import jax, jax.random as jr, jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from einops import rearrange

sys.path.append("..")
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.model.NCA_model import NCA

CHANNELS      = 8
KERNEL_STR    = ["ID", "LAP", "GRAD"]
PADDING       = "CIRCULAR"
KERNEL_SCALE  = 1
NUM_INTERVALS = 8       # number of snapshot intervals
SIZE          = 64
FIRE_RATE     = 1.0

a_true, b_true, D = 0.01, 2, 80
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

# ---------- CLI ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",  type=str, required=True, help="*.eqx file")
parser.add_argument("--mix",    type=str,
    default="0:3,1:5,2:5,3:5,4:2",
    help="comma-sep counts per IC type, e.g. '0:2,1:1'")
parser.add_argument("--outdir", type=str, default="batch_inference")
parser.add_argument("--seed",   type=int, default=123)
parser.add_argument("--time_sampling", type=int, required=True,
                    help="micro-steps the NCA advances between consecutive "
                         "comparisons with the reference images")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# ---------- parse mixture string  ---------------------------------
mix = {int(k): int(v) for k, v in
       (pair.split(":") for pair in re.split(r"[ ,]+", args.mix) if pair)}

BATCHES = sum(mix.values())
#print(f" Evaluating on {BATCHES} trajectories  →  {mix}")

# ---------- constants (identical to training) ----------------------
SIZE, CHANNELS = 64, 8
TIME_SAMPLING  = args.time_sampling
NUM_INTERVALS  = 8
a, b, D = 0.01, 2, 80
DX = 1.0

# ---------- build IC batch  ---------------------------------------
key, *sub = jr.split(jr.PRNGKey(args.seed), BATCHES + 1)
sub   = jnp.array(sub)
choices = jnp.concatenate(
    [jnp.full(n, c, dtype=jnp.int32) for c, n in mix.items()])
x0 = jax.vmap(make_ic)(sub, choices)        # (B,2,H,W)

# ---------- ground-truth PDE trajectory ---------------------------
rhs  = F_schnakenberg(PADDING="CIRCULAR", dx=DX, a=a, b=b, D=D)
vrhs = eqx.filter_vmap(rhs, in_axes=(None, 0, None), out_axes=0)
solver = PDE_solver(vrhs, 5e-3)
ts = jnp.linspace(0, TIME_SAMPLING*3, TIME_SAMPLING*NUM_INTERVALS)
T, Ypde = solver(ts, x0)                    # (T,B,2,H,W)
Ypde = rearrange(Ypde, "T B C H W -> B T C H W")
Ypde = Ypde[:, :, :1]                       # keep U
Ypde = (Ypde - Ypde.min()) / (Ypde.max() - Ypde.min())
Ypde = Ypde[:, ::TIME_SAMPLING]             # [B,T,1,H,W]

# ---------- load trained NCA --------------------------------------
dummy_nca = NCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=jax.nn.relu,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE,
    KERNEL_SCALE=KERNEL_SCALE,
    key=key,
)
nca   = eqx.tree_deserialise_leaves(args.model, dummy_nca)

def nca_rollout(x_init, k):
    def step(carry, _):
        x, key = carry
        x = nca(x, lambda z: z, key)
        return (x, jr.fold_in(key, 0)), x
    (_, _), xs = jax.lax.scan(step, (x_init, k), None, length=TIME_SAMPLING)
    return xs[-1]

v_roll = jax.vmap(nca_rollout, in_axes=(0, 0), out_axes=0)

hidden = jnp.zeros((BATCHES, CHANNELS-1, SIZE, SIZE))
x_cur  = jnp.concatenate([Ypde[:, 0], hidden], axis=1)

preds = []
k_batch = jr.split(key, BATCHES)
for _ in range(NUM_INTERVALS):
    preds.append(x_cur[:, :1])
    x_cur = v_roll(x_cur, k_batch)
preds = jnp.stack(preds, axis=1)            # (B,T,1,H,W)

# ---------- compute errors ----------------------------------------
mse = jnp.mean((preds - Ypde)**2, axis=(2,3,4))   # per-trajectory
mean_mse = float(jnp.mean(mse))
#print("\n Per-trajectory squared error:", mse)
print(f" Batch-mean squared error: {mean_mse:.4e}")

# ---------- optional visualisation of first trajectory ------------
# gt, pr = Ypde[0], preds[0]
# fig, axes = plt.subplots(3, NUM_INTERVALS, figsize=(3*NUM_INTERVALS,9))
# for t in range(NUM_INTERVALS):
#     axes[0,t].imshow(gt[t,0], origin="lower");  axes[0,t].axis("off")
#     axes[1,t].imshow(pr[t,0], origin="lower");  axes[1,t].axis("off")
#     err = (gt[t,0]-pr[t,0])**2
#     im  = axes[2,t].imshow(err, origin="lower");axes[2,t].axis("off")
# axes[0,0].set_ylabel("GT"); axes[1,0].set_ylabel("NCA"); axes[2,0].set_ylabel("Err")
# plt.colorbar(im, ax=axes[2,:], orientation="horizontal", shrink=0.7)
# plt.suptitle(f"N={BATCHES}  mean MSE={mean_mse:.3e}")
# fig.savefig(os.path.join(args.outdir, "grid.png"), dpi=150)
# plt.close(fig)
