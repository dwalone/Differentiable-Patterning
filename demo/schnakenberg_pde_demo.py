import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import time
import sys
sys.path.append('..')

from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate

# Grid and simulation parameters
SIZE = 64

a = 0.2            # feed rate parameter
b = 0.8            # removal rate parameter
D = 50.0           # diffusion coefficient for V

dx = 1.0           # spatial step size
dt = 5e-3          # time step for solver

t_max = 100.0      # total simulation time
n_steps = 200      # number of time points saved

# initialize random initial condition around homogeneous steady state
key = jr.PRNGKey(int(time.time()))
# small Gaussian perturbation
noise = jr.normal(key, shape=(2, SIZE, SIZE)) * 0.02

# schnakenberg steady state: U_eq = a + b, V_eq = b/(a + b)^2
U_eq = a + b
V_eq = b / (U_eq**2)

x0 = jnp.stack([U_eq + noise[0], V_eq + noise[1]], axis=0)

# smooth initial condition
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(3):
    x0 = op.Average(x0)

# schnakenberg RHS and numerical solver
func = F_schnakenberg(
    PADDING="CIRCULAR",
    dx=dx,
    a=a,
    b=b,
    D=D
)
solver = PDE_solver(
    func,
    dt=dt,
)

# create time vector and run solver
ts = jnp.linspace(0.0, t_max, n_steps)
ts, Y = solver(ts=ts, y0=x0)

print(f"ts.shape = {ts.shape}")
print(f"Y.shape  = {Y.shape}")

# animate only the U channel over time
my_animate(Y[:, :1], clip=False)
