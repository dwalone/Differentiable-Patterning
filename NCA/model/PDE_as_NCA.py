import jax, jax.numpy as jnp, equinox as eqx
from jaxtyping import Array, Float


class PDEasNCA(eqx.Module):
    """
    Wraps a RHS F so it behaves like an NCA cell for NCA_trainer,
    but really performs a single explicit-Euler step of the PDE:
          x ↦ x + dt · F(x)
    """
    F:  eqx.Module          # update_schnakenberg.F
    dt: float
    N_CHANNELS: int

    # dummy attributes the trainer / logger expect
    KERNEL_STR = ["ID"]
    perception = staticmethod(lambda x: x)
    N_FEATURES = 1          # any positive int is fine

    # ------------------------------------------------------------------
    # core update used inside scan
    # ------------------------------------------------------------------
    def __call__(self,
                 x: Float[Array, "{self.N_CHANNELS} h w"],
                 boundary_callback=lambda z: z,
                 key=None):
        dx = self.F(0.0, x, None)          # (t, x, args) signature of F
        return boundary_callback(x + self.dt * dx)

    # ------------------------------------------------------------------
    # minimal API required by NCA_trainer & tensorboard_log
    # ------------------------------------------------------------------
    def partition(self):
        # keeps behaviour if you later turn a,b,D into trainable arrays
        return eqx.partition(self, eqx.is_inexact_array)

    def get_config(self):
        return {"MODEL": "PDE-as-NCA", "dt": float(self.dt)}

    # ---- histogram helpers -------------------------------------------
    def get_weights(self):
        """Return dummy conv weights so logger can draw histograms."""
        C  = self.N_CHANNELS
        w1 = jnp.zeros((1, C, 1, 1), dtype=jnp.float32)   # input layer
        w2 = jnp.zeros((C, C, 1, 1), dtype=jnp.float32)   # output layer
        b2 = jnp.zeros((C, 1, 1),     dtype=jnp.float32)  # output bias
        return w1, w2, b2

    def set_weights(self, weights):
        """Trainer may call this when SPARSE_PRUNING is enabled."""
        # Neural-PDE cell has no conv weights -> ignore.
        pass

    def l1_output_weight(self):
        """Used by trainer when L1_COEFF > 0. We return 0."""
        return jnp.array(0.0, dtype=jnp.float32)

    # ---- post-training rollout for GIFs ------------------------------
    def run(self, iters, x, callback=lambda z: z, key=None):
        traj = [x]
        for i in range(iters):
            if key is not None:
                key = jax.random.fold_in(key, i)
            x = self(x, callback, key)
            traj.append(x)
        return jnp.stack(traj)
