# 6 — Variational Mode Decomposition

VMD (Dragomiretskiy & Zosso, 2014) reformulates the decomposition problem
as a variational optimisation solved by ADMM. It is non-iterative across
modes — all modes emerge simultaneously from a constrained optimisation —
which gives it much stronger noise robustness than EMD.
