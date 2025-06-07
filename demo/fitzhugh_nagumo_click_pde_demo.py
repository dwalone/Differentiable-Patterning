import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import time
import sys
sys.path.append('..')

from PDE.model.fixed_models.update_fhn import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate

# ----------------------------------------------------------------------
# 1) make_spike_ic using plain Python loops + .at
# ----------------------------------------------------------------------
def make_spike_ic(key, B, C, H, W, N_clicks=5, sigma=1.5, amplitude=1.0):
    """
    Host‐side IC builder: returns array [B,2,H,W] with random Gaussian bumps.
    """
    # precompute the 2D Gaussian kernel once
    radius = int(3 * sigma)
    xs = jnp.arange(-radius, radius + 1)
    ys = xs
    Xg, Yg = jnp.meshgrid(xs, ys, indexing='ij')
    kernel = amplitude * jnp.exp(-(Xg**2 + Yg**2) / (2 * sigma**2))

    # initialize U and V
    U0 = jnp.zeros((B, H, W), dtype=jnp.float32)
    V0 = jnp.zeros_like(U0)

    # generate all random keys upfront
    keys = jr.split(key, B * N_clicks).reshape(B, N_clicks, 2)

    # for each batch‐sample, scatter‐add N_clicks bumps
    for b in range(B):
        for n in range(N_clicks):
            subkey = keys[b, n]
            k1, k2 = jr.split(subkey)
            i = int(jr.randint(k1, (), radius, H - radius))
            j = int(jr.randint(k2, (), radius, W - radius))

            i0, i1 = i - radius, i + radius + 1
            j0, j1 = j - radius, j + radius + 1
            # scatter‐add the *entire* kernel patch
            U0 = U0.at[b, i0:i1, j0:j1].add(kernel)

    # stack into (U,V) channels
    return jnp.stack([U0, V0], axis=1)  # shape [B,2,H,W]


# ----------------------------------------------------------------------
# 2) PDE & solver parameters
# ----------------------------------------------------------------------
SIZE     = 64
dx       = 1.0
dt       = 1e-2
t_max    = 100.0
n_steps  = 200

# FitzHugh–Nagumo params
D    = 0.05         # slower inhibitor diffusion
eps_v = 0.005       # stronger timescale separation
a_v   = 0.5
a_z   = 0.0         # zero offset → excitable pulses

# 3) build random‐click IC
key = jr.PRNGKey(int(time.time()))
key, subkey = jr.split(key)
x0_batched = make_spike_ic(
    subkey,
    B=1,
    C=2,
    H=SIZE,
    W=SIZE,
    N_clicks=8,
    sigma=1.0,
    amplitude=1.0
)   # shape [1,2,64,64]

# **Strip off** that leading batch axis
x0 = jnp.squeeze(x0_batched, axis=0)   # now shape [2,64,64]

x0 = x0.astype(jnp.float64)

# slight smoothing to knock off any pixel‐aliasing
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(2):
    x0 = op.Average(x0)                # now x0 is 3-D, so Average works

# ----------------------------------------------------------------------
# 4) instantiate RHS & solver
# ----------------------------------------------------------------------
func   = F_fhn(PADDING="CIRCULAR", dx=dx, D=D, eps_v=eps_v, a_v=a_v, a_z=a_z)
solver = PDE_solver(func, dt=dt)

# ----------------------------------------------------------------------
# 5) integrate and animate
# ----------------------------------------------------------------------
ts = jnp.linspace(0.0, t_max, n_steps)
ts, Y = solver(ts=ts, y0=x0)   # Y: [n_steps, B, 2, H, W]

print("ts.shape =", ts.shape)
print("Y.shape  =", Y.shape)

# animate only the U‐channel
my_animate(Y[:, :1], clip=False)

