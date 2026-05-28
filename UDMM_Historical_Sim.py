"""
UDMM-Historical Simulator
Three-Timescale Closed System: τ_fast / τ_med / τ_slow
Mohammed Ahmed Aidaros — UDMM Research Programme

Equations:
  FAST:   ds/dt   = -∇_{g^eff} F_total + F_gen + ξ
  MEDIUM: ∂C_m/∂t = -δF_LG/δC_m + η  [GL dynamics]
  SLOW:   τ_H dH_struct/dt = -λ_H H_struct + HES(t)

  HES(t) = F₁(t) · max(0, 1 - S_eq(t))   [undischarged friction]
  F₁^crit(t) = F₁,₀ · exp(-η₁ · H_struct)
  F₂^crit(t) = F₂,₀ · exp(-η₂ · H_struct)
  C_m^int   = σ(F₁/F₁^crit(t)) · σ(F₂/F₂^crit(t))
  g^eff     = Ω(C_m, H_struct) · g⁰
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings; warnings.filterwarnings("ignore")

np.random.seed(2026)

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════
T   = 800;   dt = 0.05
t_  = np.arange(T) * dt          # max time ≈ 40

# F₀/F₁ params
alpha_abs  = 0.6
beta_diss  = 0.4
F1_max     = 3.5
F1_0_crit  = 1.6     # baseline threshold (high → hard to breach)
F2_0_crit  = 0.25

# H_struct params (slow timescale)
tau_H   = 15.0       # slow timescale constant (units: same as dt)
lam_H   = 0.02       # structural relaxation rate  (very slow)
eta_1   = 1.2        # threshold sensitivity to H_struct
eta_2   = 0.9

# Metric warping
alpha_m = 1.5
kappa_m = 2.0
beta_m  = 1.2
eps_m   = 0.01

k_sig   = 6.0        # logistic steepness

# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def sig(x, crit, k=6.0):
    return 1.0 / (1.0 + np.exp(-k * (x / (crit + 1e-8) - 1.0)))

def threshold_F1(Hs, nu=1.0):
    """F₁^crit(t) = F₁,₀ · exp(-η₁ · H_struct^ν)"""
    return F1_0_crit * np.exp(-eta_1 * Hs**nu)

def threshold_F2(Hs, nu=1.0):
    """F₂^crit(t) = F₂,₀ · exp(-η₂ · H_struct^ν)"""
    return F2_0_crit * np.exp(-eta_2 * Hs**nu)

def omega(Cm, Hs):
    """Ω(C_m, H_struct) — historical metric warping"""
    return 1.0 / ((1.0 - Cm)**alpha_m + eps_m) / (1.0 + kappa_m * Hs)**beta_m

def HES(F1, S_eq):
    """Undischarged friction → structural memory input"""
    return F1 * max(0.0, 1.0 - S_eq)

# ═══════════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════════

def simulate(F0_fn, w_fn, phi_fn, label, H0=0.0, therapy_fn=None, nu=1.0):
    """
    Returns dict with all state variables across T timesteps.
    therapy_fn(t, i, Hs) → float: additional H_struct reduction rate.
    """
    F0 = np.zeros(T); F1 = np.zeros(T); F2 = np.zeros(T)
    Cm = np.zeros(T); S  = np.zeros(T)
    Hs = np.zeros(T); HES_arr = np.zeros(T)
    F1c= np.zeros(T); F2c= np.zeros(T)
    Om = np.zeros(T); v  = np.zeros(T)
    phi = 0.0; ph = np.zeros(T); dph = np.zeros(T)

    Hs[0] = H0

    for i in range(T):
        t  = t_[i]
        n  = np.random.randn()

        # ── F₀ (fast, driven) ──────────────────────────────────────
        F0[i] = np.clip(F0_fn(t, i, n), 0, 6)
        w      = w_fn(t, i)

        # ── F₁ ODE (medium) ───────────────────────────────────────
        sat    = 1 - F1[i-1]/F1_max if i > 0 else 1.0
        prev1  = F1[i-1] if i > 0 else 0.0
        F1[i]  = np.clip(prev1 + dt*(alpha_abs*F0[i]*sat - beta_diss*prev1*w), 0, F1_max)

        # ── phi dynamics → F₂ ─────────────────────────────────────
        phi    = phi_fn(t, i, phi, n)
        ph[i]  = phi
        dph[i] = (ph[i] - ph[i-1])/dt if i > 0 else 0
        F2[i]  = np.var(dph[max(0,i-30):i+1]) if i >= 5 else 0

        # ── S_eq (stability factor) ───────────────────────────────
        denom  = alpha_abs * F0[i] / (F1[i] + 1e-8)
        S[i]   = beta_diss * w / (denom + 1e-8)

        # ── HES & H_struct ODE (slow) ─────────────────────────────
        hes_t       = HES(F1[i], S[i])
        HES_arr[i]  = hes_t
        Hs_prev     = Hs[i-1] if i > 0 else H0
        therapy_val = therapy_fn(t, i, Hs_prev) if therapy_fn else 0.0
        dHs         = (-lam_H * Hs_prev + hes_t - therapy_val) / tau_H
        Hs[i]       = np.clip(Hs_prev + dt * dHs, 0, 5.0)

        # ── History-dependent thresholds ───────────────────────────
        F1c[i] = threshold_F1(Hs[i], nu)
        F2c[i] = threshold_F2(Hs[i], nu)

        # ── C_m^int (smooth product with historical thresholds) ───
        Cm[i]  = sig(F1[i], F1c[i], k_sig) * sig(F2[i], F2c[i], k_sig)

        # ── Metric warping ─────────────────────────────────────────
        Om[i]  = omega(Cm[i], Hs[i])

    v = np.abs(np.gradient(F1, dt))
    return dict(F0=F0, F1=F1, F2=F2, Cm=Cm, S=S,
                Hs=Hs, HES=HES_arr, F1c=F1c, F2c=F2c,
                Om=Om, v=v, phi=ph, label=label)


# ═══════════════════════════════════════════════════════════════════
# EIGHT SCENARIOS
# ═══════════════════════════════════════════════════════════════════

scenarios = []

# 1. HEALTHY — moderate friction, good narrative, H_struct stays low
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.35 + 0.12*np.sin(t*.5) + 0.04*n,
    w_fn     = lambda t,i:   0.45,
    phi_fn   = lambda t,i,p,n: p + dt*(0.25*np.sin(t*.4) - 0.4*p + 0.05*n),
    label    = "Healthy", H0=0.05))

# 2. OCD — low F0, poor dissolution → H_struct accumulates slowly
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.18 + 0.04*n,
    w_fn     = lambda t,i:   0.04,
    phi_fn   = lambda t,i,p,n: p + dt*(0.6*np.sin(t*1.2) - 0.25*p + 0.03*n),
    label    = "OCD", H0=0.3))

# 3. ANXIETY — high F0 spikes, oscillating
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.8 + 0.5*np.sin(t*2.0+n*0.4) + 0.15*abs(n),
    w_fn     = lambda t,i:   0.18 + 0.08*np.sin(t*.6),
    phi_fn   = lambda t,i,p,n: p + dt*(1.2*np.sin(t*3.0+n) - 0.15*p + 0.3*n),
    label    = "Anxiety", H0=0.2))

# 4. DEPRESSION — low F0, narrative absent, H_struct high from history
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.08 + 0.01*n,
    w_fn     = lambda t,i:   0.015,
    phi_fn   = lambda t,i,p,n: p + dt*(0.02 - 0.5*p),
    label    = "Depression", H0=1.8))   # historical load

# 5. EARLY TRAUMA — H_struct very high from start, thresholds permanently low
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.3 + 0.15*np.sin(t*.4) + 0.06*n,
    w_fn     = lambda t,i:   0.15,
    phi_fn   = lambda t,i,p,n: p + dt*(0.4*np.sin(t*.8) - 0.3*p + 0.08*n),
    label    = "Early Trauma", H0=2.5, nu=1.5))  # catastrophic ν

# 6. ACUTE TRAUMA — F0 spike then slow recovery (with high H_struct residue)
def F0_acute(t, i, n):
    if 3 < t < 7: return 4.5 + 0.5*abs(n)
    elif 7 < t < 15: return 1.5 - 0.12*(t-7) + 0.2*abs(n)
    else: return 0.2 + 0.05*n
scenarios.append(simulate(
    F0_fn    = F0_acute,
    w_fn     = lambda t,i: 0.04 if t < 15 else 0.12,
    phi_fn   = lambda t,i,p,n: p + dt*((1.5*np.sin(t*4)*(1 if 3<t<12 else 0)) - 0.3*p + 0.15*n),
    label    = "Acute Trauma", H0=0.1))

# 7. THERAPY (starting from Depression H0=1.8)
def therapy_fn(t, i, Hs):
    """Relational therapy: μ·Ψ_eff·H_struct, starts at t=8.
    Note: multiplied by tau_H to yield intended effective rate
    after division by tau_H in the H_struct ODE (Eq. 16)."""
    if t > 8:
        psi = 0.25 + 0.1*np.sin(t*0.3)   # relational resonance
        return 0.15 * tau_H * psi * Hs    # compensated: effective μ=0.15
    return 0.0
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: (0.08 + 0.01*n) if t <= 8 else (0.08 + 0.01*n + 0.04 + (t-8)*0.005),
    w_fn     = lambda t,i:   0.015 + (0.12 if t>8 else 0),
    phi_fn   = lambda t,i,p,n: p + dt*(0.02 - 0.5*p + (0.1*np.sin(t*0.6) if t>8 else 0)),
    label    = "Therapy\n(Depression→)", H0=1.8, therapy_fn=therapy_fn))

# 8. HYSTERESIS DEMO — EXACT same F0 AND w as Healthy, only H₀ differs
# This produces a PURE path-dependence proof: the only difference is history
scenarios.append(simulate(
    F0_fn    = lambda t,i,n: 0.35 + 0.12*np.sin(t*.5) + 0.04*n,
    w_fn     = lambda t,i:   0.45,   # identical to Healthy — pure hysteresis proof
    phi_fn   = lambda t,i,p,n: p + dt*(0.25*np.sin(t*.4) - 0.4*p + 0.05*n),
    label    = "Same F₀,\nHigh H₀", H0=2.0))

COLS = ["#1E8449","#8E44AD","#D35400","#2C3E50",
        "#C0392B","#1F618D","#1ABC9C","#F39C12"]
NAMES = [s["label"] for s in scenarios]

# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#080C14")
gs = gridspec.GridSpec(5, 4, figure=fig,
                       hspace=0.46, wspace=0.32,
                       left=0.05, right=0.97,
                       top=0.94, bottom=0.03)
W = "#ECF0F1"

def sax(ax, title, fs=9):
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors="#BDC3C7", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#2C3E50")
    ax.set_title(title, color=W, fontsize=fs, fontweight="bold", pad=6)

# ── ROW 0: H_struct all scenarios ──────────────────────────────────
ax = fig.add_subplot(gs[0, :])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax.plot(t_, s["Hs"], color=c, lw=1.8, label=lbl, alpha=0.92)
ax.axhline(0.5, color="#E74C3C", ls=":", lw=0.9, alpha=0.5, label="Low threshold (H=0.5)")
ax.axhline(1.5, color="#E74C3C", ls="--", lw=0.9, alpha=0.5, label="Critical (H=1.5)")
ax.fill_between(t_, 1.5, 5, alpha=0.06, color="#E74C3C")
ax.set_ylabel("H_struct (structural memory)", color="#BDC3C7", fontsize=9)
ax.legend(fontsize=7.5, facecolor="#0D1117", labelcolor=W,
          framealpha=0.7, ncol=8, loc="upper left")
sax(ax, "τ_H · dH_struct/dt = −λ_H·H_struct + HES(t)  |  Slow Timescale (τ_slow)  [UDMM-Historical]", fs=10)

# ── ROW 1: History-dependent thresholds ───────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax.plot(t_, s["F1c"], color=c, lw=1.5, label=lbl, alpha=0.85)
ax.axhline(F1_0_crit, color=W, ls="--", lw=0.8, alpha=0.4, label=f"F₁,₀ baseline={F1_0_crit}")
ax.set_ylabel("F₁^crit(t) = F₁,₀·exp(−η₁·H_struct)", color="#BDC3C7", fontsize=8)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=8)
ax.legend(fontsize=7, facecolor="#0D1117", labelcolor=W, framealpha=0.6, ncol=4)
sax(ax, "History-Dependent Threshold F₁^crit  [Structural Brittleness]")

# ── ROW 1: Metric warping Ω ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 2:4])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax.plot(t_, np.clip(s["Om"], 0, 30), color=c, lw=1.5,
            label=lbl, alpha=0.85)
ax.axhline(1.0, color=W, ls=":", lw=0.7, alpha=0.4)
ax.set_ylabel("Ω(C_m, H_struct)", color="#BDC3C7", fontsize=8)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=8)
ax.set_ylim(0, 25)
ax.legend(fontsize=7, facecolor="#0D1117", labelcolor=W, framealpha=0.6, ncol=4)
sax(ax, "Metric Warping Ω(C_m, H_struct)  [Historical + Instantaneous]")

# ── ROW 2: HES input + C_m^int ────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax.plot(t_, s["HES"], color=c, lw=1.4, label=lbl, alpha=0.8)
ax.set_ylabel("HES = F₁·max(0, 1−S_eq)", color="#BDC3C7", fontsize=8)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=8)
ax.legend(fontsize=7, facecolor="#0D1117", labelcolor=W, framealpha=0.6, ncol=4)
sax(ax, "HES(t) — Undischarged Friction Feeding H_struct")

ax = fig.add_subplot(gs[2, 2:4])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax.plot(t_, s["Cm"], color=c, lw=1.5, label=lbl, alpha=0.88)
ax.axhline(0.5, color=W, ls=":", lw=0.7, alpha=0.35)
ax.set_ylabel("C_m^int = σ(F₁/F₁^crit)·σ(F₂/F₂^crit)", color="#BDC3C7", fontsize=8)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=8)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=7, facecolor="#0D1117", labelcolor=W, framealpha=0.6, ncol=4)
sax(ax, "C_m^int  [Historical thresholds make same F₁ → different C_m]")

# ── ROW 3: Hysteresis demo + therapy ──────────────────────────────
ax = fig.add_subplot(gs[3, 0:2])
# Healthy vs High-H0 with same F0
ax.plot(t_, scenarios[0]["Cm"], color=COLS[0], lw=2.0, label="Healthy (H₀=0.05)")
ax.plot(t_, scenarios[7]["Cm"], color=COLS[7], lw=2.0, ls="--",
        label="Same F₀, High H₀=2.0")
ax.fill_between(t_,
    scenarios[0]["Cm"], scenarios[7]["Cm"],
    alpha=0.15, color="#E74C3C", label="Hysteresis gap")
ax.set_ylabel("C_m^int", color="#BDC3C7", fontsize=9)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=9)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=8.5, facecolor="#0D1117", labelcolor=W, framealpha=0.7)
sax(ax, "Hysteresis: Same F₀ Input → Different C_m (Path Dependence)", fs=9.5)

ax = fig.add_subplot(gs[3, 2:4])
# Therapy scenario
dep = scenarios[3]; ther = scenarios[6]
ax.plot(t_, dep["Hs"],  color=COLS[3], lw=1.8, label="Depression (no therapy)")
ax.plot(t_, ther["Hs"], color=COLS[6], lw=2.0, label="Depression + Therapy (t>8)")
ax.axvline(8, color=W, ls="--", lw=0.9, alpha=0.5, label="Therapy start")
ax.fill_between(t_,
    dep["Hs"], ther["Hs"],
    where=t_>8, alpha=0.2, color=COLS[6], label="ΔH_struct from therapy")
ax.set_ylabel("H_struct", color="#BDC3C7", fontsize=9)
ax.set_xlabel("Time", color="#BDC3C7", fontsize=9)
ax.legend(fontsize=8.5, facecolor="#0D1117", labelcolor=W, framealpha=0.7)
sax(ax, "Therapy as dH_struct/dt < 0  [ΔH ∝ −μ·Ψ_eff·H_struct]", fs=9.5)

# ── ROW 4: 3D phase portrait + summary table ─────────────────────
ax4a = fig.add_subplot(gs[4, 0:2])
for s, c in zip(scenarios, COLS):
    lbl = s["label"].replace("\n", " ")
    ax4a.scatter(s["F1"][::5], s["Hs"][::5], c=c, s=5, alpha=0.3)
    ax4a.scatter(s["F1"][-1], s["Hs"][-1], c=c, s=100,
                 marker="D", edgecolors=W, lw=0.8,
                 label=f"{lbl}", zorder=10)

# threshold curve: F1 at which C_m = 0.5 as function of H_struct
hs_range = np.linspace(0, 3, 100)
F1_at_threshold = F1_0_crit * np.exp(-eta_1 * hs_range)
ax4a.plot(F1_at_threshold, hs_range, color="#E74C3C", lw=1.5,
          ls="--", label="C_m=0.5 frontier\n(upper bound, F\u2082\u2248sat.)", zorder=8)
ax4a.fill_betweenx(hs_range, F1_at_threshold, F1_max,
                   alpha=0.07, color="#E74C3C")

ax4a.set_xlabel("F₁ (accumulated friction)", color="#BDC3C7", fontsize=9)
ax4a.set_ylabel("H_struct (structural memory)", color="#BDC3C7", fontsize=9)
ax4a.legend(fontsize=7.5, facecolor="#0D1117", labelcolor=W,
            framealpha=0.7, ncol=2, loc="upper right")
sax(ax4a, "Phase Portrait (F₁, H_struct)  [Red = C_m threshold frontier]")

# Summary stats
ax4b = fig.add_subplot(gs[4, 2:4])
ax4b.axis("off")
summary = [["State", "H_struct", "F₁^crit", "C_m", "S_eq"]]
for s in scenarios:
    name = s["label"].replace("\n", " ")
    summary.append([
        name,
        f"{s['Hs'].mean():.2f}",
        f"{s['F1c'].mean():.2f}",
        f"{s['Cm'].mean():.2f}",
        f"{np.clip(s['S'], 0, 5).mean():.2f}"
    ])
tbl = ax4b.table(cellText=summary[1:], colLabels=summary[0],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#0D1117" if r > 0 else "#1F3864")
    cell.set_text_props(color=W)
    cell.set_edgecolor("#2C3E50")
ax4b.set_title("Summary Statistics (time-averaged)",
               color=W, fontsize=9, fontweight="bold", pad=8)

fig.text(0.5, 0.97,
    "UDMM-Historical  |  τ_H · dH_struct/dt = −λ_H·H_struct + HES(t)  |  "
    "Path-Dependent Cognitive Geometry  |  UDMM Research Programme",
    ha="center", va="top", color=W, fontsize=11, fontweight="bold")

plt.savefig("/mnt/user-data/outputs/UDMM_Historical.png",
            dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Figure saved: UDMM_Historical.png")

# ── Final summary ─────────────────────────────────────────────────
print(f"\n{'State':<20}{'H_struct':>10}{'F₁^crit':>10}"
      f"{'C_m':>8}{'HES':>8}{'S_eq':>8}")
print("─" * 66)
for s in scenarios:
    name = s["label"].replace("\n", " ")
    print(f"{name:<20}{s['Hs'].mean():>10.3f}"
          f"{s['F1c'].mean():>10.3f}"
          f"{s['Cm'].mean():>8.3f}"
          f"{s['HES'].mean():>8.3f}"
          f"{np.clip(s['S'],0,5).mean():>8.3f}")
