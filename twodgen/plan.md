# Experiment Plan — NPJ 2D (SCDM-aligned) — v0.1
**Project:** twodgen (2D slab, pbc_mask=1,1,0)  
**Goal:** paper-ready evidence for npj Computational Materials  
**Reference:** SCDM-style symmetry-constrained 2D generation  
**Owner:** <your name>  
**Created:** 2026-02-10

---

## 0. Non-negotiables (freeze these before running)
### 0.1 Sampling budgets & seeds
- Quick runs: N = 2,000 per setting
- Final runs: N = 20,000 per setting
- Seeds: {0, 1, 2} (min 3 seeds; report mean±std)

### 0.2 Artifact standard (every run must output)
- `run_metadata.json` (commit hash, configs, seed, model ckpt, thresholds)
- `projection_stats.json` (trigger counts, magnitudes)
- `metrics_summary.json` (all KPIs)
- `failure_breakdown.json` (geom + energy taxonomies)
- `plots/` (auto figures)
- `samples/` (optional: CIF/POSCAR for top candidates)

### 0.3 Output directory convention (mandatory)
`runs/<EXPERIMENT_ID>/<YYYYMMDD_HHMMSS>/...`

---

## 1. Frozen metric definitions (KPIs)
### 1.1 Validity (geometry)
A sample is geom-valid if all pass:
- collision-free (min_dist >= __ Å)
- cross-vacuum-risk-free (slab constraint: __)
- in-plane degeneracy-free (inplane Gram cond <= __)
- volume/lattice sanity bounds (volume in [__, __])
- optional: angle bounds in [__, __]

Report:
- `success_geom_rate`
- `collision_rate`
- `cross_vacuum_risk_rate`
- `inplane_degen_rate`
- `bad_volume_rate`
- `post_project_trigger_any_rate`

### 1.2 Symmetry
A sample is symmetry-valid if:
- spglib succeeds
- space group matches target (conditional)

Report:
- `spacegroup_match_rate`
- `spglib_fail_rate`
- `symmetry_violation_breakdown` (typed)

### 1.3 Screening & stability
Report:
- `mlip_relax_success_rate`
- energy distribution (median, Q1/Q3)
- DFT spot-check pass rate (among top-K)

### 1.4 Novelty & diversity
Report:
- novelty distance to train (fingerprint/local-env)
- diversity coverage across:
  - space group bins
  - composition family bins
  - N_atoms bins
  - lattice bins
- QD curve: validity vs diversity

---

## 2. Target thresholds (paper-ready goals)
### 2.1 Validity targets
- `success_geom_rate` >= **0.65** (final N=20k, 3-seed mean)
- `inplane_degen_rate` <= **0.05**
- `bad_volume_rate` <= **0.10**

### 2.2 Symmetry targets (conditional)
- `spacegroup_match_rate` >= **0.85**
- `spglib_fail_rate` <= **0.05**

### 2.3 Discovery targets
- Provide funnel counts at N>=20k (preferably 100k if compute allows)
- DFT spot-check: K = 20–100 (diversity-aware)
- Report “novel & stable” counts under declared thresholds

### 2.4 Diversity constraint (anti-collapse)
- Diversity metrics must not degrade significantly when validity improves
- Must provide validity–diversity tradeoff plot

---

## 3. Experiment Matrix (do these in order)
> Each experiment lists Purpose → Settings → Command stub → Outputs → Pass/Fail → Paper usage

### E0 — Protocol Sanity (must run once)
**Purpose:** validate metric pipeline & artifact outputs  
**Settings:** tiny N=200, seed=0, one checkpoint  
**Outputs:** ensure all JSON + plots produced  
**Pass:** no missing fields; metrics computed  
**Paper usage:** none (infrastructure)

---

### E1 — Validity Ablations (Projection components)
#### E1.1 Baseline vs Full Projection
**Purpose:** quantify major validity jump and failure reduction  
**Settings:**
- A: projection OFF
- B: projection ON (cond+angle+volume)
Fixed ckpt, fixed steps, same N/seeds
**Command stub:**
- `python -m twodgen.eval --config configs/bench/E1_1.yaml --seed 0`
- repeat seeds 1,2
**Pass:**
- `success_geom_rate(B) - success_geom_rate(A) >= +0.15`
- `bad_volume_rate` significantly reduced
**Paper:** Fig.2, Fig.3, Table 1

#### E1.2 Component Ablation
**Purpose:** identify which guard contributes most + QD effects  
**Settings:** {cond only, angle only, volume only, cond+angle, cond+angle+volume}  
**Pass:** choose default guard set based on validity & diversity balance  
**Paper:** Fig.2, Fig.6, Table 1

#### E1.3 g_scale Sweep
**Purpose:** robustness; avoid volume blow-ups or over-projection  
**Settings:** g_scale in {0.5, 1.0, 1.5} (adjust to your actual range)  
**Pass:** select g_scale with best validity–diversity tradeoff  
**Paper:** Appendix S2 (optional main)

---

### E2 — Training vs Sampling Synergy
#### E2.1 Curriculum Loss Schedule Ablation
**Purpose:** show model learns constraints; projection trigger rate decreases  
**Settings:** different ramp schedules; repulsion on/off  
**Pass:** comparable validity with lower trigger rate; improved generalization  
**Paper:** Appendix + small figure in main if strong

---

### E3 — Symmetry Controllability (SCDM-aligned)
#### E3.1 SG Conditional: Soft vs Hard Consistency
**Purpose:** justify symmetry contribution; show controllability metrics  
**Settings:**
- soft: symmetry loss only
- hard: crystal-family lattice hard constraint + symmetry projection (if implemented)
**Metrics:** match rate, spglib fail, violation breakdown  
**Pass:** match >=0.85, spglib_fail <=0.05  
**Paper:** Fig.4, Table 2

#### E3.2 (Optional, high-impact) Wyckoff-level constraint
**Purpose:** strongest alignment with SCDM  
**Pass:** improves symmetry metrics at comparable diversity  
**Paper:** Main if achieved; else future work

---

### E4 — Screening Loop (MLIP → DFT)
#### E4.1 MLIP-Scale Screening Stats
**Purpose:** discovery evidence with large-N statistics  
**Pipeline:**
1) generate N = 20k (final), optionally 100k (strong)
2) geom gate
3) MLIP relax (CHGNet)
4) select top-K with diversity-aware sampling
**Outputs:** `screening.csv`, top candidates CIF  
**Pass:** clear funnel stats; meaningful low-energy population  
**Paper:** Fig.5, Table 3

#### E4.2 DFT Spot-check
**Purpose:** credibility for “stable” claim  
**K:** 20–100  
**Pass:** report how many remain stable after DFT relax; discuss MLIP bias  
**Paper:** Table 3 + case studies

---

### E5 — Novelty & Diversity (must-have)
#### E5.1 Novelty & De-dup
**Purpose:** ensure validity gains are not memorization  
**Method:** fingerprint distance; clustering; de-dup rule  
**Pass:** novelty comparable across settings  
**Paper:** Table 3, Fig.6

#### E5.2 Diversity Coverage & QD Tradeoff
**Purpose:** defend against mode collapse critique  
**Method:** coverage across bins; plot validity vs diversity  
**Pass:** validity improves without collapsing diversity  
**Paper:** Fig.6

---

## 4. Paper mapping (what goes where)
- **Intro:** motivate 2D + symmetry + discovery loop
- **Method:** projection operator + symmetry module + screening loop
- **Results 5.1:** E1 (validity)
- **Results 5.3:** E3 (symmetry)
- **Results 5.4:** E4 (screening)
- **Results 5.5:** E5 (novelty/diversity)

---

## 5. Checklist for “paper-ready” completion
- [ ] Table 1: validity metrics (mean±std, N=20k, 3 seeds)
- [ ] Fig.2: ablation heatmap (projection combos)
- [ ] Fig.3: failure taxonomy stacked bars (+ trigger rate trend)
- [ ] Table 2 + Fig.4: symmetry controllability + violation breakdown
- [ ] Fig.5 + Table 3: screening funnel + DFT spot-check summary
- [ ] Fig.6: validity–diversity tradeoff + novelty stats
- [ ] Supplement: S1–S6 fully reproducible artifacts
