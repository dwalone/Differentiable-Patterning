import jax
import jax.numpy as jnp
import equinox as eqx
import time
import sys
sys.path.append('..')

from PDE.model.fixed_models.update_fhn import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate

# 1) Grid & Domain parameters
SIZE = 64             # 64×64 grid
dx   = 1.0            # grid spacing
L    = SIZE * dx      # domain size in each direction

# 2) FitzHugh–Nagumo parameters
D       = 0.1         # diffusion coefficient for v
eps_v   = 0.01        # time‐scale separation for v
a_v     = 0.5         # v‐recovery rate
a_z     = 0.1         # v‐offset

# 3) Solver timestepping
dt    = 1e-2          # Euler time‐step (stable if dt < dx^2/(4*max(1,D)))
t_max = 100.0         # total simulated time
n_steps = 200         # number of saved snapshots

# 4) Build analytic cosine‐mode IC for Turing–Hopf
m = 4                # number of half‐waves across the domain
# create physical coords x,y in [0,L)
xs = jnp.linspace(0, L, SIZE, endpoint=False)
ys = xs
X, Y = jnp.meshgrid(xs, ys, indexing="ij")

# u0 = cos(mπx/L) cos(mπy/L),   v0 = 0
u0 = jnp.cos(m * jnp.pi * X / L) * jnp.cos(m * jnp.pi * Y / L)
v0 = jnp.zeros_like(u0)
x0 = jnp.stack([u0, v0], axis=0)   # shape [2,H,W]

# 5) (Optional) slight smoothing to avoid checkerboard modes
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(2):
    x0 = op.Average(x0)

# 6) Instantiate RHS module & solver
func   = F_fhn(PADDING="CIRCULAR", dx=dx,
               D=D, eps_v=eps_v, a_v=a_v, a_z=a_z)
solver = PDE_solver(func, dt=dt)   # fixed‐step Euler semidiscrete

# 7) Integrate and save snapshots
ts = jnp.linspace(0.0, t_max, n_steps)
ts, Y = solver(ts=ts, y0=x0)      # Y: [n_steps, 2, H, W]

print(f"ts.shape = {ts.shape}")     # -> (200,)
print(f"Y.shape  = {Y.shape}")      # -> (200, 2, 64, 64)

# 8) Animate only the u‐component (channel 0)
my_animate(Y[:, :1], clip=False)
