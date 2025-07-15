import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops

class F(eqx.Module):
    ops: Ops
    a: float
    b: float
    D: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 a=0.014,
                 b=0.054,
                 D=2.0):
        self.a = a
        self.b = b
        self.D = D
        self.ops = Ops(PADDING, dx, KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar, "2 x y"],
                 args) -> Float[Scalar, "2 x y"]:
        # Split concentration fields
        U = X[0:1]
        V = X[1:2]

        # Reaction-diffusion kinetics
        dU = self.ops.Lap(U) + U**2 * V - U * (self.a+self.b)
        dV = self.D * self.ops.Lap(V) - U**2 * V + self.a * (1 - V)

        # Concatenate back to a 2-channel field
        return jnp.concatenate((dU, dV), axis=0)
