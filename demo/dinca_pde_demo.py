import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.append('..')

from PDE.model.fixed_models.update_dinca import F as F_dinca
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate

# Grid and simulation parameters
SIZE = 64

dx = 1.0           # spatial step size
dt = 5e-3         # time step for solver

t_max = 100.0      # total simulation time
n_steps = 200      # number of time points saved

# === DIFFUSION FEATURES (physical units) ===
diffusion_weights = {
    # ΔChannel 0
    'dx_ch0_ch0': +0.00000e+00,
    'dx_ch0_ch1': +0.00000e+00,
    'dy_ch0_ch0': +0.00000e+00,
    'dy_ch0_ch1': +0.00000e+00,
    'lap_ch0_ch0': +0.00000e+00,
    'lap_ch0_ch1': +0.00000e+00,
    # ΔChannel 1
    'dx_ch1_ch0': +0.00000e+00,
    'dx_ch1_ch1': +0.00000e+00,
    'dy_ch1_ch0': +0.00000e+00,
    'dy_ch1_ch1': +0.00000e+00,
    'lap_ch1_ch0': +0.00000e+00,
    'lap_ch1_ch1': +0.00000e+00,
}


# === REACTION TERMS (physical units) ===
reaction_weights = {
    # ΔChannel 0
    'u_ch0': +0.00000e+00,
    'v_ch0': +1.45873e+00,
    'uu_ch0': +0.00000e+00,
    'uv_ch0': +0.00000e+00,
    'vv_ch0': +0.00000e+00,
    'uuu_ch0': +1.74262e-02,
    'uuv_ch0': +0.00000e+00,
    'uvv_ch0': +0.00000e+00,
    'vvv_ch0': +0.00000e+00,
    # ΔChannel 1
    'u_ch1': -2.88293e-03,
    'v_ch1': -4.42155e-02,
    'uu_ch1': +0.00000e+00,
    'uv_ch1': +0.00000e+00,
    'vv_ch1': +0.00000e+00,
    'uuu_ch1': +0.00000e+00,
    'uuv_ch1': +0.00000e+00,
    'uvv_ch1': +0.00000e+00,
    'vvv_ch1': +0.00000e+00,
}


# === CONSTANT SOURCES (bias) ===
a = +0.00000e+00  # source term for u-equation
b = +2.28497e-01  # source term for v-equation

# initialize random initial condition around zero equilibrium
# recompute the true Schnakenberg steady-state
U_eq = a + b
V_eq = b / ( (a + b)**2 )

# initialize around that
key = jr.PRNGKey(int(time.time()))
noise = jr.normal(key, shape=(2, SIZE, SIZE)) * 0.02
x0 = jnp.stack([U_eq + noise[0], V_eq + noise[1]], axis=0)

# smooth initial condition
op = Ops(PADDING="CIRCULAR", dx=dx, KERNEL_SCALE=3)
for _ in range(3):
    x0 = op.Average(x0)

# build the DINCA PDE RHS and numerical solver
func = F_dinca(
    PADDING="CIRCULAR",
    dx=dx,
    diffusion_weights=diffusion_weights,
    reaction_weights=reaction_weights,
    a=a,
    b=b
)
solver = PDE_solver(func, dt=dt)

# create time vector and run solver
ts = jnp.linspace(0.0, t_max, n_steps)
ts, Y = solver(ts=ts, y0=x0)

print(f"ts.shape = {ts.shape}")
print(f"Y.shape  = {Y.shape}")

# Get global min/max for consistent color mapping
vmin = Y[:, 0].min()
vmax = Y[:, 0].max()

fig, ax = plt.subplots()
im = ax.imshow(Y[0, 0], cmap='viridis', vmin=vmin, vmax=vmax)

def update(frame):
    im.set_array(Y[frame, 0])
    ax.set_title(f"t = {ts[frame]:.2f}")
    return [im]

ani = FuncAnimation(fig, update, frames=len(ts), interval=50, blit=True)

ani.save("schnakenberg_simulation.mp4", writer="ffmpeg", fps=20)
print("Saved animation to schnakenberg_simulation.mp4")
