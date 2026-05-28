# UDMM Simulation Code
## The Geometry of Cognitive Constraint: A Ginzburg-Landau Field Theory of Psychological Phase Transitions

**Mohammed Ahmed Aidaros**
Independent Researcher, Atbara, Sudan
ORCID: 0009-0005-1948-402X

---

## Overview

This repository contains the simulation code accompanying the manuscript:

> Aidaros, M. A. (2026). The geometry of cognitive constraint: A Ginzburg-Landau field theory
> of psychological phase transitions. *Manuscript submitted for publication.*

Two simulation modules are provided:

| File | Description | Figure in paper |
|---|---|---|
| `UDMM_Historical_Sim.py` | Three-timescale system with H_struct structural memory | Figure 1 |
| `UDMM_Friction_Spectrum_Sim.py` | F₀/F₁/F₂ friction spectrum across six clinical states | Figure 2 |

---

## Requirements

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.11
```

Install with:
```bash
pip install numpy matplotlib scipy
```

No proprietary software required. Tested on Python 3.10-3.12.

---

## Usage

### Figure 1 — Three-Timescale Historical Simulation

```bash
python UDMM_Historical_Sim.py
```

Produces `UDMM_Historical.png` with 5 panel rows:
- Row 1: H_struct dynamics (slow timescale τ_slow) — 8 clinical scenarios
- Row 2: History-dependent thresholds F₁^crit(t) and metric warping Ω
- Row 3: HES input and C_m^int
- Row 4: Hysteresis demonstration + therapy trajectory
- Row 5: Phase portrait (F₁, H_struct) with C_m = 0.5 frontier

### Figure 2 — Friction Spectrum

```bash
python UDMM_Friction_Spectrum_Sim.py
```

Produces `UDMM_Friction_Spectrum.png` with:
- F₁ dynamics across 6 scenarios
- Individual F₀/F₁/C_m profiles per scenario
- Phase portrait (F₁, F₂)
- Smooth joint-threshold surface C_m^int = σ(F₁)·σ(F₂)

---

## Core Equations Implemented

### Three-timescale closed system (Eqs. 21-26 in manuscript)

**Fast (τ_fast):**
```
ds/dt = -∇_{g^eff} F_total + F_gen + ξ
```

**Medium (τ_med — GL constraint field):**
```
∂C_m/∂t = κ Δ_{g^eff} C_m - dV/dC_m + η(s,t)
```

**Slow (τ_slow — structural memory):**
```
τ_H · dH_struct/dt = -λ_H · H_struct + HES(t)
```

where `HES(t) = F₁(t) · max(0, 1 - S_eq(t))` — undischarged friction (Eq. 17).

### History-dependent thresholds (Eqs. 19-20)
```
F₁^crit(t) = F₁,₀^crit · exp(-η₁ · H_struct(t)^ν)
F₂^crit(t) = F₂,₀^crit · exp(-η₂ · H_struct(t)^ν)
```

### Smooth joint-threshold C_m (Eq. 25)
```
C_m^int = σ(F₁/F₁^crit(t)) · σ(F₂/F₂^crit(t))
where σ(x) = 1/(1 + exp(-k(x-1)))
```

### Historical metric warping (Eq. 26)
```
g^eff_ij(s,t) = Ω(C_m, H_struct) · g^(0)_ij(s)
Ω = 1/[(1-C_m)^α + ε] · 1/[1 + κ·H_struct]^β
```

---

## Eight Clinical Scenarios (UDMM_Historical_Sim.py)

| # | Label | H₀ | Key feature |
|---|---|---|---|
| 1 | Healthy | 0.05 | S_eq > 1; thresholds intact |
| 2 | OCD | 0.30 | High F₁, low F₂; rigid but moving |
| 3 | Anxiety | 0.20 | High F₂; oscillating attractors |
| 4 | Depression | 1.80 | Frozen: low F₀, v_path ≈ 0 |
| 5 | Early Trauma | 2.50 | F₁^crit ≈ 0.004; permanent brittleness |
| 6 | Acute Trauma | 0.10 | F₀ spike → H_struct surge |
| 7 | Therapy | 1.80 | H_struct decreasing via relational resonance |
| 8 | Hysteresis demo | 2.00 | Same F₀ as Healthy; 5× metric warping |

---

## Parameter Configuration

All parameters are defined at the top of each script and documented inline.
Core parameters and their derivation sources are listed in Appendix B, Table B.1 of the manuscript.

To reproduce exact figures from the paper, use `np.random.seed(2026)` (already set in both scripts).

---

## Correspondence

Mohammed Ahmed Aidaros
ORCID: 0009-0005-1948-402X
Atbara, Sudan

---

## Citation

If you use this code, please cite:

```bibtex
@article{aidaros2026geometry,
  title   = {The Geometry of Cognitive Constraint: A Ginzburg-Landau Field Theory
             of Psychological Phase Transitions},
  author  = {Aidaros, Mohammed Ahmed},
  year    = {2026},
  note    = {Manuscript submitted for publication},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

MIT License. Copyright (c) 2026 Mohammed Ahmed Aidaros.
