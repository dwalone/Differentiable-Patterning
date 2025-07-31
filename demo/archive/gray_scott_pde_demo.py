import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import time
import sys
sys.path.append('..')
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate


SIZE = 64 # Grid size




key = jr.PRNGKey(int(time.time())) # JAX PRNG key generation
x0 = jr.uniform(key,shape=(2,SIZE,SIZE))


# Make smoothly random initial conditions
op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
for i in range(5):
    x0 = op.Average(x0)
x0 = x0.at[0].set(jnp.where(x0[0]>0.51,0.0,1.0))
x0 = op.Average(x0)
x0 = x0.at[1].set(1-x0[0])



# Define RHS of PDE
func = F_gray_scott(PADDING="CIRCULAR",    # Boundary condition, choose from PERIODIC, ZEROS, REFLECT, REPLICATE
                    dx=0.5,                # Spatial stepsize - scales spatial patterns, but also affects stability
                    alpha=0.0623,          # Parameters that affect patterning 
                    gamma=0.06268)         # 
solver = PDE_solver(func,dt=0.2)           # Wrap the RHS in a numerical solver
T,Y = solver(ts=jnp.linspace(0,10000,100),y0=x0) # Generate solution trajectory

print(T.shape)
print(Y.shape)
my_animate(Y[:,:1],clip=False)