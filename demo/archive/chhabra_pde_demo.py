import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import time
import sys
sys.path.append('..')
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.utils import my_animate


SIZE = 64 # Grid size

key = jr.PRNGKey(int(time.time())) # JAX PRNG key generation
scale=0.5
x0 = jr.uniform(key,shape=(2,SIZE,SIZE))*scale

# Define RHS of PDE
func = F_chhabra(PADDING="CIRCULAR",    # Boundary condition, choose from PERIODIC, ZEROS, REFLECT, REPLICATE
                 dx=0.5,                # Spatial stepsize - scales spatial patterns, but also affects stability
                 DA=0.0025)             # Diffusion of activator signal - controls what kind of patterning happens
solver = PDE_solver(func,dt=0.2)        # Wrap the RHS in a numerical solver
T,Y = solver(ts=jnp.linspace(0,10000,100),y0=x0) # Generate solution trajectory

print(T.shape)
print(Y.shape)
my_animate(Y[:,:1],clip=False)

