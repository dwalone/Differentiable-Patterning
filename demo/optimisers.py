from typing import Sequence
import optax
import jax
import jax.numpy as jnp
import equinox as eqx

optimiser = None
# -------------------------------------------------------------------------
# --------------------------- Masking Optimiser ---------------------------
# -------------------------------------------------------------------------
def masked_optimiser(iters, lr, nca, keep_diff, keep_reac, bias_mask):
    def build_weight_mask_fhn(model,
                            keep_reac_labels: dict[int, list[str]] = None,
                            keep_diff_indices: Sequence[tuple[int,int]] = None):
        """
        Returns a jnp.bool_ mask of shape [C_out, N_features] where only
        the specified diffusion indices *and* the specified reaction labels
        (per output channel) are True.
        """
        C = model.N_CHANNELS
        F = model.N_FEATURES
        K_diff = 3 * C             # your index offset into the feature vector
        labels = model.reaction_labels

        # start with everything frozen
        mask = jnp.zeros((C, F), dtype=bool)

        # 1) diffusion: very similar
        if keep_diff_indices is not None:
            for c_out, feat_idx in keep_diff_indices:
                mask = mask.at[c_out, feat_idx].set(True)

        # 2) reaction: *per* output channel
        if keep_reac_labels is not None:
            for c_out, label_list in keep_reac_labels.items():
                for i, lbl in enumerate(labels):
                    if lbl in label_list:
                        mask = mask.at[c_out, K_diff + i].set(True)

        return mask
    # Get params
    nca_diff, _ = nca.partition()
    # Define a PyTree mask
    def make_optax_mask(nca, weight_mask):
        """
        Convert a [C, F] weight mask into a PyTree mask that matches nca_diff
        """
        def mask_fn(param):
            if param is nca.layers[0].weight:
                return weight_mask[:, :, None, None].astype(bool)
            elif param is nca.layers[0].bias:
                return jnp.ones_like(param, dtype=bool)  # trainable bias
            else:
                return False  # freeze everything else
        return jax.tree_util.tree_map(mask_fn, nca_diff)
    mask = build_weight_mask_fhn(
        nca,
        keep_reac_labels=keep_reac,
        keep_diff_indices=keep_diff
    )
    bias_mask = jnp.array(bias_mask)[:, None, None]
    # Now build a PyTree of masks exactly matching your params
    def make_tree_mask(params):
        def mask_fn(param):
            # conv weights come in as shape (2,15,1,1)
            if param.shape == mask.shape + (1,1):
                return mask[:, :, None, None]
            # conv bias has shape (2,1,1)
            elif param.shape == bias_mask.shape:
                return bias_mask
            # everything else: either train fully or freeze fully
            else:
                return jnp.ones_like(param, dtype=bool)
        return jax.tree_util.tree_map(mask_fn, params)
    tree_mask = make_tree_mask(nca_diff)

    # --- ZERO-OUT ALL FROZEN PARAMETERS ON THE MODEL ---
    # conv weights: shape (C, F, 1, 1), mask has shape (C, F)
    w = nca.layers[0].weight * mask[:, :, None, None]
    nca.layers[0] = eqx.tree_at(lambda m: m.weight, nca.layers[0], w)
    # conv biases: shape (C, 1, 1), bias_mask was already expanded
    b = nca.layers[0].bias * bias_mask
    nca.layers[0] = eqx.tree_at(lambda m: m.bias,   nca.layers[0], b)
    # — now every weight/bias outside your keep_* lists is zeroed and will stay frozen —

    # Custom gradient‐masking transform
    def zero_out_updates(updates, state, params=None):
        masked = jax.tree_util.tree_map(lambda g, m: g * m, updates, tree_mask)
        return masked, state
    masking = optax.GradientTransformation(
        init=lambda _: (),
        update=zero_out_updates
    )
    schedule     = optax.exponential_decay(lr, transition_steps=iters, decay_rate=0.99)
    base_optim = optax.chain(
        optax.scale_by_param_block_norm(),
        optax.nadam(learning_rate=schedule)
    )

    optimiser = optax.chain(
        #optax.clip_by_global_norm(1.0),
        base_optim,
        masking
    )
    return optimiser
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --------------------------- Normal Optimiser ----------------------------
# -------------------------------------------------------------------------
def normal_optimiser(iters, lr, dr):
    schedule  = optax.exponential_decay(lr, transition_steps=iters, decay_rate=dr)
    optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))
    return optimiser
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --------------------------- Warmup Optimiser ----------------------------
# -------------------------------------------------------------------------
def warmup_optimiser(iters, lr, warmup_steps=400):
    peak_lr      = lr
    decay_steps  = iters - warmup_steps
    warmup = optax.linear_schedule(
        init_value=1e-7,      # start very small
        end_value=peak_lr,
        transition_steps=warmup_steps
    )
    decay = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=decay_steps
    )
    schedule = optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps]
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),   # optional: clip huge gradients
        optax.adamw(learning_rate=schedule, weight_decay=1e-5)
    )
    return optimiser
# -------------------------------------------------------------------------