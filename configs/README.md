# Experiment configs

Training and benchmark settings live here—**no separate `models/` folder** in this public repo.

| Subfolder | Role |
|-----------|------|
| **`custom/`** | Baseline and CoSA-related custom configs (`baseline/`, `cosa_v3/`). |
| **`opencd/`** | Open-CD–style Python configs: `_base_/`, per-model folders (`fcsn/`, `stanet/`, `bit/`, …), and `common/` schedules. |

These files are the **source of truth** for hyperparameters, backbones, and dataset entry points when you plug into Open-CD (or your fork). Dataset and checkpoint paths must be set to **your** machine.
