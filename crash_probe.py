print("imp‑einops");        from einops import rearrange
print("imp‑ops");           from Common.model.spatial_operators import Ops
print("imp‑schnak");        from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
print("imp‑solver");        from PDE.model.solver.semidiscrete_solver import PDE_solver
print("imp‑trainer");       from NCA.trainer.NCA_trainer import NCA_Trainer      # ← imports jaxpruner
print("imp‑augment");       from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
print("imp‑nca");           from NCA.model.NCA_model import NCA
print("after‑imports")
