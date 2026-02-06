# MLIP Finetune Report (Slab / 2D)
**Date**: 2026-02-06
**Status**: template (no training yet)

This file is a *report template* for the optional Workstream C in
`twodgen/plan_next_sampling_energy_mlip.md`. The goal is to make the finetune
decision (go/no-go) reproducible and auditable, even if we decide not to train.

## 0. Summary (Go/No-Go)
**Decision**: TBD (go/no-go)

**Go/No-Go criteria (pass any 2):**
- relax_success_rate improves
- force_MAE decreases
- success_energy_rate improves and `fail_reason_energy` shifts away from `non_converge/nan_*`
- runtime/cost does not significantly worsen

## 1. Baseline
- Baseline MLIP: CHGNet (version: TBD)
- Baseline checkpoint id/path: TBD
- Inference device/dtype: TBD
- Eval pipeline version: `twodgen/evaluate/eval_samples.py` (energy taxonomy enabled)

## 2. Data
### 2.1 Dataset definition
- Structures source:
  - [ ] JDfT2D
  - [ ] C2DB slabs
  - [ ] sampled outputs (specify run dirs)
- Labels:
  - [ ] forces
  - [ ] energies
  - [ ] stresses (optional)

### 2.2 Split
- seed: TBD
- split files / manifest:
  - train: TBD
  - val: TBD
  - test: TBD

### 2.3 Filters / constraints
- geometry gate (if used): success_geom only
- vacuum policy: TBD
- element whitelist/blacklist: TBD

## 3. Training Configuration
- optimizer: TBD
- lr schedule: TBD
- batch size: TBD
- epochs / steps: TBD
- gradient clipping: TBD
- loss weights: force/energy/stress = TBD
- mixed precision: TBD

## 4. Evaluation Protocol
### 4.1 Metrics (val/test)
- force_MAE: TBD
- energy_MAE: TBD
- relax_success_rate: TBD
- nan_rate (energy/force): TBD
- wall time:
  - per structure inference: TBD
  - per structure relax: TBD

### 4.2 Energy-chain regression (twodgen)
Run the same scripts used in phase2:
- `bash twodgen/scrip/eval_with_energy.sh` (with finetuned model wired in)
- Compare against baseline CHGNet:
  - `energy_available_rate`
  - `success_energy_rate`
  - `fail_reason_energy_counts`

## 5. Results
### 5.1 Table
| Model | force_MAE (val) | force_MAE (test) | energy_MAE (test) | relax_success_rate (test) | nan_rate | success_energy_rate (twodgen) | Notes |
|------|------------------|------------------|-------------------|---------------------------|----------|-------------------------------|-------|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | |
| finetune | TBD | TBD | TBD | TBD | TBD | TBD | |

### 5.2 Fail reason shift
- before: TBD
- after: TBD

## 6. Artifacts / Repro
- config: `twodgen/mlip_finetune_config.yaml`
- registry: `twodgen/model_registry.json`
- training logs dir: TBD
- evaluation outputs dir: TBD

## 7. Commands Run
```bash
# record the exact commands used
```

