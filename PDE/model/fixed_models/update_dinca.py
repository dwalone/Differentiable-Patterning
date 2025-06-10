import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops

class F(eqx.Module):
    """
    General PDE with learned diffusion, reaction, and source weights (DINCA).
    """
    ops: Ops
    diffusion_weights: dict
    reaction_weights: dict
    a: float
    b: float

    def __init__(
        self,
        PADDING: str,
        dx: float,
        diffusion_weights: dict,
        reaction_weights: dict,
        a: float = 0.0,
        b: float = 0.0,
        KERNEL_SCALE: int = 1
    ):
        """
        Args:
            PADDING (str): padding mode for convolution (e.g. 'CIRCULAR' or 'ZERO')
            dx (float): grid spacing
            diffusion_weights (dict): weights for spatial derivatives, keys:
                'dx_ch0_ch0', 'dx_ch0_ch1', 'dy_ch0_ch0', 'dy_ch0_ch1',
                'lap_ch0_ch0', 'lap_ch0_ch1', 'dx_ch1_ch0', 'dx_ch1_ch1',
                'dy_ch1_ch0', 'dy_ch1_ch1', 'lap_ch1_ch0', 'lap_ch1_ch1'
            reaction_weights (dict): weights for reaction monomials, keys:
                'u_ch0', 'v_ch0', 'uu_ch0', 'uv_ch0', 'vv_ch0',
                'uuu_ch0', 'uuv_ch0', 'uvv_ch0', 'vvv_ch0',
                'u_ch1', 'v_ch1', 'uu_ch1', 'uv_ch1', 'vv_ch1',
                'uuu_ch1', 'uuv_ch1', 'uvv_ch1', 'vvv_ch1'
            a (float): constant source term for channel 0
            b (float): constant source term for channel 1
            KERNEL_SCALE (int): scale factor for convolution kernel
        """
        self.ops = Ops(PADDING, dx, KERNEL_SCALE)
        self.diffusion_weights = diffusion_weights
        self.reaction_weights = reaction_weights
        self.a = a
        self.b = b

    def __call__(
        self,
        t: Float[Scalar, ""],
        X: Float[Scalar, "2 x y"],
        args: PyTree
    ) -> Float[Scalar, "2 x y"]:
        # Split concentration fields
        U = X[0:1]
        V = X[1:2]

        # === Diffusion contributions ===
        # compute gradients and Laplacians
        gr_U = self.ops.Grad(U)       # shape: (2, C, x, y)
        gr_V = self.ops.Grad(V)
        lap_U = self.ops.Lap(U)       # shape: (C, x, y)
        lap_V = self.ops.Lap(V)

        # Channel 0 diffusion
        dU_diff = (
            self.diffusion_weights['dx_ch0_ch0'] * gr_U[0]
            + self.diffusion_weights['dx_ch0_ch1'] * gr_V[0]
            + self.diffusion_weights['dy_ch0_ch0'] * gr_U[1]
            + self.diffusion_weights['dy_ch0_ch1'] * gr_V[1]
            + self.diffusion_weights['lap_ch0_ch0'] * lap_U
            + self.diffusion_weights['lap_ch0_ch1'] * lap_V
        )
        # Channel 1 diffusion
        dV_diff = (
            self.diffusion_weights['dx_ch1_ch0'] * gr_U[0]
            + self.diffusion_weights['dx_ch1_ch1'] * gr_V[0]
            + self.diffusion_weights['dy_ch1_ch0'] * gr_U[1]
            + self.diffusion_weights['dy_ch1_ch1'] * gr_V[1]
            + self.diffusion_weights['lap_ch1_ch0'] * lap_U
            + self.diffusion_weights['lap_ch1_ch1'] * lap_V
        )

        # === Reaction contributions ===
        # Channel 0
        dU_react = (
            self.reaction_weights['u_ch0'] * U
            + self.reaction_weights['v_ch0'] * V
            + self.reaction_weights['uu_ch0'] * U**2
            + self.reaction_weights['uv_ch0'] * (U * V)
            + self.reaction_weights['vv_ch0'] * V**2
            + self.reaction_weights['uuu_ch0'] * U**3
            + self.reaction_weights['uuv_ch0'] * (U**2 * V)
            + self.reaction_weights['uvv_ch0'] * (U * V**2)
            + self.reaction_weights['vvv_ch0'] * V**3
        )
        # Channel 1
        dV_react = (
            self.reaction_weights['u_ch1'] * U
            + self.reaction_weights['v_ch1'] * V
            + self.reaction_weights['uu_ch1'] * U**2
            + self.reaction_weights['uv_ch1'] * (U * V)
            + self.reaction_weights['vv_ch1'] * V**2
            + self.reaction_weights['uuu_ch1'] * U**3
            + self.reaction_weights['uuv_ch1'] * (U**2 * V)
            + self.reaction_weights['uvv_ch1'] * (U * V**2)
            + self.reaction_weights['vvv_ch1'] * V**3
        )

        # === Constant sources ===
        dU_bias = self.a
        dV_bias = self.b

        # Total time derivatives
        dU = dU_diff + dU_react + dU_bias
        dV = dV_diff + dV_react + dV_bias

        # Concatenate back to 2-channel field
        return jnp.concatenate((dU, dV), axis=0)
