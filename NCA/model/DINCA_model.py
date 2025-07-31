# ================================================================
# DINCA_model.py – Dynamics‑Identification Neural Cellular Automaton
# ================================================================
"""A drop‑in replacement for `NCA_model` that learns an explicit
polynomial reaction term plus diffusion operators.  The public API –
`__call__`, `perception`, `get_weights`, `set_weights`, `partition`,
`l1_output_weight`, and `pretty_print_pde` – is identical to the one
expected by `NCA_trainer`, tensor‑board utilities, and pruning code.
"""

from __future__ import annotations

import time
from itertools import product
from typing import List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from Common.model.abstract_model import AbstractModel
from Common.model.spatial_operators import Ops

# ------------------------------------------------------------------
# helper functions --------------------------------------------------
# ------------------------------------------------------------------

def _monomial_exponents(num_channels: int, degree: int = 3) -> List[Tuple[int, ...]]:
    """All exponent tuples with 0 < Σe ≤ *degree*."""
    exps: List[Tuple[int, ...]] = []
    for total_deg in range(1, degree + 1):
        for exp in product(range(total_deg + 1), repeat=num_channels):
            if 0 < sum(exp) == total_deg:
                exps.append(exp)
    return exps


def _apply_monomial(x: Float[Array, "C H W"], exp: Sequence[int]):
    out = jnp.ones_like(x[0])
    for ch, e in enumerate(exp):
        if e:
            out = out * (x[ch] ** e)
    return out


def _exp_to_label(exp: Sequence[int], var_names: Sequence[str]):
    return "".join(v * e for v, e in zip(var_names, exp) if e) or "1"

# ------------------------------------------------------------------
# DINCA class -------------------------------------------------------
# ------------------------------------------------------------------

class DINCA(AbstractModel):
    """Polynomial‑reaction Neural Cellular Automaton."""

    # static metadata fields (excluded from parameter PyTree) --------
    reaction_labels: List[str]      = eqx.static_field()
    feature_names:  List[str]       = eqx.static_field()
    N_FEATURES:     int             = eqx.static_field()
    KERNEL_STR:     List[str]       = eqx.static_field()
    _monomial_exps: List[Tuple[int, ...]] = eqx.static_field()

    # parameter fields ----------------------------------------------
    N_CHANNELS: int
    FIRE_RATE:  float
    op:         Ops
    linear:     eqx.nn.Conv2d

    # ---------------------------------------------------------------
    def __init__(
        self,
        N_CHANNELS: int,
        FIRE_RATE: float = 1.0,
        degree: int = 3,
        PADDING: str = "CIRCULAR",
        KERNEL_SCALE: int = 1,
        key: Key | None = None,
    ) -> None:
        super().__init__()
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))

        self.N_CHANNELS = N_CHANNELS
        self.FIRE_RATE  = FIRE_RATE
        self.op = Ops(PADDING=PADDING, dx=1, KERNEL_SCALE=KERNEL_SCALE)

        # --- construct feature list --------------------------------
        grad_names = [f"gradx_{c}" for c in range(N_CHANNELS)] + \
                     [f"grady_{c}" for c in range(N_CHANNELS)] + \
                     [f"lap_{c}"   for c in range(N_CHANNELS)]

        var_names = ["u", "v", "w", "z", "q", "r"][:N_CHANNELS]
        self._monomial_exps = _monomial_exponents(N_CHANNELS, degree)
        self.reaction_labels = [_exp_to_label(e, var_names) for e in self._monomial_exps]

        self.feature_names = grad_names + self.reaction_labels
        self.N_FEATURES   = len(self.feature_names)
        self.KERNEL_STR   = ["GRADX", "GRADY", "LAP"] + self.reaction_labels

        # --- 1×1 Conv initialised to zero ---------------------------
        k_w, _ = jax.random.split(key)
        self.linear = eqx.nn.Conv2d(
            in_channels=self.N_FEATURES,
            out_channels=N_CHANNELS,
            kernel_size=1,
            use_bias=True,
            key=k_w,
        )
        self.linear = eqx.tree_at(lambda l: l.weight, self.linear, jnp.zeros_like(self.linear.weight))
        self.linear = eqx.tree_at(lambda l: l.bias,   self.linear, jnp.zeros_like(self.linear.bias))

    # ------------------------------------------------------------------
    # feature maps ------------------------------------------------------
    def perception(self, x: Float[Array, "C H W"], /):
        gx, gy = self.op.Grad(x)
        lap    = self.op.Lap(x)
        return jnp.concatenate([gx, gy, lap], axis=0)  # (3C, H, W)

    def _reaction(self, x: Float[Array, "C H W"], /):
        return jnp.stack([_apply_monomial(x, e) for e in self._monomial_exps], axis=0)

    # ------------------------------------------------------------------
    # forward step ------------------------------------------------------
    def __call__(
        self,
        x: Float[Array, "C H W"],
        boundary_callback=lambda z: z,
        key: Key = jax.random.PRNGKey(0),
    ) -> Float[Array, "C H W"]:
        features = jnp.concatenate([self.perception(x), self._reaction(x)], axis=0)
        dx = self.linear(features)
        if self.FIRE_RATE < 1.0:
            dx *= jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        x_new = jnp.clip(x + dx, -0.5, 1.5)
        return boundary_callback(x_new)

    # ------------------------------------------------------------------
    # helpers for trainer/logger ---------------------------------------
    def get_weights(self):
        """Return `[W, b]` list for pruning/logging."""
        return [self.linear.weight, self.linear.bias]

    def set_weights(self, w):
        new_linear = eqx.tree_at(lambda l: l.weight, self.linear, w[0])
        new_linear = eqx.tree_at(lambda l: l.bias,   new_linear, w[1])
        object.__setattr__(self, "linear", new_linear)

    def partition(self):
        return eqx.partition(self, eqx.is_inexact_array)

    def l1_output_weight(self):
        """Scalar ℓ¹ penalty used by the trainer."""
        return jnp.sum(jnp.abs(self.linear.weight))

    # pretty‑print ------------------------------------------------------
    def pretty_print_pde(self, thresh: float = 1e-3) -> str:
        W = jnp.squeeze(self.linear.weight)  # (C, F)
        b = jnp.squeeze(self.linear.bias)    # (C,)
        rows = []
        for c in range(self.N_CHANNELS):
            terms: List[str] = []
            for f, name in enumerate(self.feature_names):
                coeff = float(W[c, f])
                if abs(coeff) < thresh:
                    continue
                if abs(coeff - 1.0) < 1e-6:
                    terms.append(name)
                elif abs(coeff + 1.0) < 1e-6:
                    terms.append(f"- {name}")
                else:
                    terms.append(f"{coeff:+.3g} {name}")
            if abs(float(b[c])) >= thresh:
                terms.append(f"{float(b[c]):+.3g}")
            rhs = " + ".join(terms) if terms else "0"
            rows.append(f"∂x{c}/∂t = {rhs}")
        return "\n".join(rows)

    # config for wandb --------------------------------------------------
    def get_config(self):
        return {
            "MODEL":       "DINCA",
            "N_CHANNELS":  self.N_CHANNELS,
            "FIRE_RATE":   self.FIRE_RATE,
            "FEATURES":    self.feature_names,
        }
