import jax.numpy as jnp

def read_out(trainer, range_u, range_v, dt):
    # ============================ read‑out ===============================
    _, w_raw, b_raw = trainer.NCA_model.get_weights()
    b_raw = jnp.squeeze(b_raw) 
    w_raw = jnp.squeeze(w_raw)  # (C_out, F)
    C = trainer.NCA_model.N_CHANNELS
    K_diff = 3 * C
    w_diff_raw = w_raw[:, :K_diff]
    w_reac_raw = w_raw[:, K_diff:]

    # ---------- helper: label → exponents --------------------------------
    reac_labels = trainer.NCA_model.reaction_labels

    def label_to_exponents(lbl: str):
        lbl = lbl.strip()
        return lbl.count('u'), lbl.count('v')  # power of u, power of v

    # ---------- rescale to physical units ---------------------------------
    scale_uv = jnp.array([range_u, range_v])

    # diffusion / gradient (same formula for all spatial derivatives)
    w_diff_phys = jnp.zeros_like(w_diff_raw)
    for j, deriv in enumerate(["dx","dy","lap"]):
        for v in range(C):
            idx = j*C + v
            w_diff_phys = w_diff_phys.at[:, idx].set(
                w_diff_raw[:, idx] 
                / scale_uv[v]     # ← divide, not multiply
                / dt
            )

    # reaction monomials
    w_reac_phys = jnp.zeros_like(w_reac_raw)
    for k, lbl in enumerate(reac_labels):
        pu, pv = label_to_exponents(lbl)
        denom = (scale_uv[0]**pu) * (scale_uv[1]**pv) * dt
        w_reac_phys = w_reac_phys.at[:, k].set(
            w_reac_raw[:, k] / denom
        )

    # biases
    bias_phys = b_raw / dt        # bias has no range scaling (p=q=0)


    # biases (constant sources)
    bias_phys = b_raw / dt  # channel‑wise

    # ---------- pretty print in dictionary format ------------------------

    diff_names = ["dx", "dy", "lap"]
    diff_labels = [f"{d}(ch{c})" for d in diff_names for c in range(C)]

    print("\n# === DIFFUSION FEATURES (physical units) ===")
    print("diffusion_weights = {")
    for c_out in range(C):
        print(f"    # ΔChannel {c_out}")
        for i, d in enumerate(diff_names):
            for c_in in range(C):
                label = f"{d}_ch{c_out}_ch{c_in}"
                idx = i * C + c_in
                val = float(w_diff_phys[c_out, idx])
                print(f"    '{label}': {val:+.5e},")
    print("}")

    print("\n\n# === REACTION TERMS (physical units) ===")
    print("reaction_weights = {")
    for c_out in range(C):
        print(f"    # ΔChannel {c_out}")
        for i, lbl in enumerate(reac_labels):
            label = f"{lbl}_ch{c_out}"
            val = float(w_reac_phys[c_out, i])
            print(f"    '{label}': {val:+.5e},")
    print("}")

    print("\n\n# === CONSTANT SOURCES (bias) ===")
    print(f"a = {float(bias_phys[0]):+.5e}  # source term for u-equation")
    print(f"b = {float(bias_phys[1]):+.5e}  # source term for v-equation")