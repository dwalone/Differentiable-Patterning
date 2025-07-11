import jax
import jax.numpy as jnp
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar
from Common.model.abstract_model import AbstractModel  # handle save/load/partition
from Common.model.spatial_operators import Ops         # Sobel + Laplacian kernels
from Common.model.custom_functions import construct_polynomials, construct_polynomials_with_labels
from einops import rearrange


class NCA_DINCA(AbstractModel):
    # --- public attributes expected elsewhere in the code -----------------
    layers: list                            # list[eqx.Module | callable]
    KERNEL_STR: list                        # for logging & loss masks
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable                   # for external use by Trainer
    MAX_POLY_DEGREE: int = eqx.field(static=True)
    _REACTION_FEATURES: int = eqx.field(static=True)  # <-- Add this
    L1_COEFF: float
    reaction_labels: list

    # ---------------------------------------------------------------------
    def __init__(
        self,
        N_CHANNELS: int,
        MAX_POLY_DEGREE: int = 3,
        PADDING: str = "CIRCULAR",
        FIRE_RATE: float = 1.0,
        KERNEL_SCALE: int = 1,
        key: Key = jax.random.PRNGKey(int(time.time())),
        KERNEL_STR: list = [
            "ID",
            "GRAD",
            "LAP",
        ],
        L1_COEFF: float = 1e-2,
        reaction_labels: list = []
    ):
        super().__init__()  # nothing to initialise in AbstractModel

        self.N_CHANNELS = int(N_CHANNELS)
        self.FIRE_RATE = float(FIRE_RATE)
        self.KERNEL_STR = KERNEL_STR
        self.MAX_POLY_DEGREE = int(MAX_POLY_DEGREE)
        self.op = Ops(
            PADDING=PADDING,
            dx=1.0,
            KERNEL_SCALE=KERNEL_SCALE,
            SMOOTHING=1.0,
        )
        self.L1_COEFF = float(L1_COEFF)

        # ------------------------------------------------ perception stack
        _perception_channels = 3 * self.N_CHANNELS  # sobel_x, sobel_y, lap

        def _perception(X: Float[Array, "C x y"]) -> Float[Array, "F x y"]:
            """Stack [∂x, ∂y, ∇²] features per channel – no gradients here."""
            grad = self.op.Grad(X)         # (2, C, x, y)
            lap = self.op.Lap(X)           # (C, x, y)
            return rearrange(
                [grad[0], grad[1], lap],  # list length = 3
                "k C x y -> (k C) x y",
            )

        self.perception = _perception  # exposed exactly like in vanilla NCA

        # ------------------------------------------------ reaction stack size
        poly_dummy, labels = construct_polynomials_with_labels(
            jnp.zeros(self.N_CHANNELS),
            self.MAX_POLY_DEGREE,
            var_names=["u", "v"]
        )
        self._REACTION_FEATURES = poly_dummy.shape[0]
        self.reaction_labels = labels

        # total feature depth fed into the 1×1 convolution
        self.N_FEATURES = _perception_channels + self._REACTION_FEATURES
        #------------------------------------------------ linear update rule
        key_w, key_b = jax.random.split(key, 2)
        linear = eqx.nn.Conv2d(
            in_channels=self.N_FEATURES,
            out_channels=self.N_CHANNELS,
            kernel_size=1,
            use_bias=True,
            key=key_w,
        )
        # initialise weights ≈ 0 so early training is stable
        w_zeros = jnp.zeros((self.N_CHANNELS, self.N_FEATURES, 1, 1))
        b_zeros = jnp.zeros((self.N_CHANNELS, 1, 1))
        linear = eqx.tree_at(lambda l: l.weight, linear, w_zeros)
        linear = eqx.tree_at(lambda l: l.bias,   linear, b_zeros)
        # store in a *list* so APIs relying on ``layers`` do not crash
        self.layers = [linear]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _build_reaction_features(
        X: Float[Array, "C x y"],
        max_power: int,
    ) -> Float[Array, "K x y"]:
        C, H, W = X.shape
        X_flat = rearrange(X, "C h w -> (h w) C")  # (P, C)
        # vmap over the pixel axis
        poly_flat = jax.vmap(lambda v: construct_polynomials(v, max_power))(X_flat)
        K = poly_flat.shape[-1]
        return rearrange(poly_flat, "(h w) k -> k h w", h=H), K

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        x: Float[Array, "{self.N_CHANNELS} x y"],
        boundary_callback=lambda z: z,
        key: Key = jax.random.PRNGKey(int(time.time())),
    ) -> Float[Array, "{self.N_CHANNELS} x y"]:
        """Run one DINCA step (callable identical to vanilla NCA)."""

        # 1) diffusion‑like features (fixed filters)
        feats_diff = self.perception(x)                           # (3*C, H, W)

        # 2) reaction‑like features (monomials up to cubic)
        feats_reac, _K = self._build_reaction_features(x, self.MAX_POLY_DEGREE)

        # 3) concatenate along feature/channel axis
        feats = jnp.concatenate([feats_diff, feats_reac], axis=0)  # (F, H, W)

        # 4) 1×1 convolution → residual Δx
        dx = self.layers[0](feats)

        # 5) stochastic update (same semantics as vanilla NCA)
        sigma = jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        x_new = x + sigma * dx
        x_new = jnp.clip(x_new, -0.5, 1.5) 
        return boundary_callback(x_new)

    # ---------------------------------------------------------------- get/set
    def get_config(self):
        return {
            "MODEL": "NCA_DINCA",
            "N_CHANNELS": self.N_CHANNELS,
            "KERNEL_STR": self.KERNEL_STR,
            "MAX_POLY_DEGREE": self.MAX_POLY_DEGREE,
            "PADDING": self.op.PADDING,
            "FIRE_RATE": self.FIRE_RATE,
        }

    # ------------------------- compatibility helpers ---------------------
    def set_weights(self, weights):
        if len(weights) == 2:
            w, b = weights
        elif len(weights) >= 3:
            _, w, b = weights[:3]
        # Directly set without reshaping
        self.layers[0] = eqx.tree_at(lambda l: l.weight, self.layers[0], w)
        self.layers[0] = eqx.tree_at(lambda l: l.bias, self.layers[0], b)

    def get_weights(self):
        diff, _ = self.partition()
        ws, _ = jax.tree_util.tree_flatten(diff)
        w1 = ws[0]  # Keep 4D shape (out, in, 1, 1)
        b1 = ws[1]  # Keep 1D shape (out,)
        dummy_w0 = jnp.zeros_like(w1)  # Preserve shape
        return [dummy_w0, w1, b1]

    def run(self,
            iters: Int[Scalar, ""],
            x: Float[Array, "{self.N_CHANNELS} x y"],
            callback=lambda x:x,
            key: Key =jax.random.PRNGKey(int(time.time())))->Float[Array,"{iters} {self.N_CHANNELS} x y"]:
        
        trajectory = []
        trajectory.append(x)
        for i in range(iters):
            key = jax.random.fold_in(key,i)
            x = self(x,callback,key=key)
            trajectory.append(x)
        return jnp.array(trajectory)

    # ---------- PATCH 2: L1 helper ---------------------------------------
    def l1_output_weight(self):
        """Mean |w| of the 1×1 kernel (for regularisation)."""
        return jnp.mean(jnp.concatenate([
            jnp.abs(self.layers[0].weight).ravel(),
            jnp.abs(self.layers[0].bias).ravel()
        ]))

    # --------------------------------------------------------------------

