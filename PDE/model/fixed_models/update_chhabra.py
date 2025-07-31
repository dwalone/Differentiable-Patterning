import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops
class F(eqx.Module):
    ops: Ops
    KA: float
    KI: float
    KdA: float
    KdI: float
    SA: float
    SI: float
    DA: float
    DI: float
    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 KA=0.0,
                 KI=1.0,
                 KdA=0.001,
                 KdI=0.008,
                 SA=0.01,
                 SI=0.01,
                 DA=0.0025, # 0.0025 or 0.014
                 DI=0.4
                 ):
        """Implementation of basic pattern formation model from Chhabra et al 2019

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): _description_
            logistic_growth_rate (float, optional): _description_. Defaults to 0.1.
            gamma (float, optional): _description_. Defaults to 10.0.
            alpha (float, optional): _description_. Defaults to 0.5.
            chi (float, optional): _description_. Defaults to 5.0.
            D (float, optional): _description_. Defaults to 0.1.
        """
        
        self.KA = KA
        self.KI = KI
        self.KdA = KdA
        self.KdI = KdI
        self.SA = SA
        self.SI = SI
        self.DA = DA
        self.DI = DI    
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:
        
        A = X[0:1]
        I = X[1:2]
        
        dA = self.DA*self.ops.Lap(A) + self.SA*A*A / (self.KI + I) - self.KdA*A#*(1+self.KA*A*A)) - self.KdA*A
        dI = self.DI*self.ops.Lap(I) + self.SI*A*A - self.KdI*I
        
        return jnp.concatenate((dA,dI),axis=0)
    
