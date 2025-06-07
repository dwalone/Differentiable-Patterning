from NCA.trainer.NCA_trainer import NCA_Trainer
from jaxtyping import Float,Array,Key
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from einops import reduce,rearrange
import datetime
import Common.trainer.loss as loss

class NCA_Trainer_multiscale_loss(NCA_Trainer):
    def __init__(self,
                 LOSS_SCALES,
                 *args,
                 **kwargs):
        super().__init__(*args,**kwargs)
        self.LOSS_SCALES = LOSS_SCALES

    def loss_func(self,
                 x:Float[Array, "N CHANNELS x y"],
                 y:Float[Array, "N CHANNELS x y"],
                 key:Key)->Float[Array, "N"]:
        
        Xs = [reduce(x,"N CHANNELS (x Dx) (y Dy)->N CHANNELS x y",Dx=d,Dy=d,reduction="mean") for d in self.LOSS_SCALES]
        Ys = [reduce(y,"N CHANNELS (x Dx) (y Dy)->N CHANNELS x y",Dx=d,Dy=d,reduction="mean") for d in self.LOSS_SCALES]
        
        losses = jnp.array([self._loss_func(X,Y,key,8) for X,Y in zip(Xs,Ys)])
        return jnp.mean(losses, axis=0)  # average over scales
        