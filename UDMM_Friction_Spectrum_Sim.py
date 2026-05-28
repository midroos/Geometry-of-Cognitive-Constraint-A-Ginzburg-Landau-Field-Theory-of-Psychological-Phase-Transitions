"""
UDMM Friction Spectrum Simulator — Calibrated v2
F0/F1/F2 Dynamic Hierarchy — Six Clinical Patterns
Mohammed Ahmed Aidaros — UDMM Research Programme
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
pass
import warnings; warnings.filterwarnings("ignore")

np.random.seed(42)

T  = 600; dt = 0.05
t_ = np.arange(T) * dt

# ── Calibrated thresholds ─────────────────────────────────────────────────
alpha = 0.6; beta = 0.4
F1_max  = 3.5; F1_crit = 1.6
F2_crit = 0.25          # calibrated so anxiety/psychosis exceed it clearly
k_sig   = 6.0

def sig(x, crit, k=6.0):
    return 1/(1+np.exp(-k*(x/crit-1)))

def simulate(F0_fn, w_fn, phi_fn, label):
    F0=np.zeros(T); F1=np.zeros(T); F2=np.zeros(T)
    Cm=np.zeros(T); S=np.zeros(T)
    phi=0.0; ph=np.zeros(T); dph=np.zeros(T)
    for i in range(T):
        t=t_[i]; noise=np.random.randn()
        F0[i]=np.clip(F0_fn(t,i,noise),0,5)
        w=w_fn(t,i)
        # F1 ODE
        sat=1-F1[i-1]/F1_max if i>0 else 1.0
        prev=F1[i-1] if i>0 else 0.0
        F1[i]=np.clip(prev+dt*(alpha*F0[i]*sat - beta*prev*w),0,F1_max)
        # phi
        phi=phi_fn(t,i,phi,noise)
        ph[i]=phi
        dph[i]=(ph[i]-ph[i-1])/dt if i>0 else 0
        # F2: rolling variance of dphi (window=30)
        F2[i]=np.var(dph[max(0,i-30):i+1]) if i>=5 else 0
        # C_m smooth product
        Cm[i]=sig(F1[i],F1_crit,k_sig)*sig(F2[i],F2_crit,k_sig)
        # S_eq stationary
        denom=alpha*F0[i]/(F1[i]+1e-6)
        S[i]=beta*w/(denom+1e-6)
    v=np.abs(np.gradient(F1,dt))
    return dict(F0=F0,F1=F1,F2=F2,Cm=Cm,S=S,v=v,phi=ph,label=label)

# ── Six scenarios ─────────────────────────────────────────────────────────
sc=[]

# 1. HEALTHY — moderate F0, good narrative, stable phi
sc.append(simulate(
    F0_fn =lambda t,i,n: 0.35+0.12*np.sin(t*.5)+0.04*n,
    w_fn  =lambda t,i: 0.45,
    phi_fn=lambda t,i,p,n: p+dt*(0.25*np.sin(t*.4)-0.4*p+0.05*n),
    label ="Healthy"))

# 2. OCD — low F0, F1 saturates (poor dissolution), phi rigid oscillation, v>0
sc.append(simulate(
    F0_fn =lambda t,i,n: 0.18+0.04*n,
    w_fn  =lambda t,i: 0.04,      # narrative almost inactive
    phi_fn=lambda t,i,p,n: p+dt*(0.6*np.sin(t*1.2)-0.25*p+0.03*n),
    label ="OCD"))

# 3. ANXIETY — high F0 spikes, F2 elevated (dphi oscillates rapidly)
sc.append(simulate(
    F0_fn =lambda t,i,n: 0.8+0.5*np.sin(t*2.0+n*0.4)+0.15*np.abs(n),
    w_fn  =lambda t,i: 0.18+0.08*np.sin(t*.6),
    phi_fn=lambda t,i,p,n: p+dt*(1.2*np.sin(t*3.0+n)-0.15*p+0.3*n),
    label ="Anxiety"))

# 4. DEPRESSION — low F0, F1 high, phi frozen, v ≈ 0
sc.append(simulate(
    F0_fn =lambda t,i,n: 0.08+0.01*n,
    w_fn  =lambda t,i: 0.015,     # narrative absent
    phi_fn=lambda t,i,p,n: p+dt*(0.02-0.5*p),   # collapses to 0
    label ="Depression"))

# 5. PSYCHOSIS — oscillating F0, high F2 (phi chaotic), C_m fragmented
sc.append(simulate(
    F0_fn =lambda t,i,n: (1.2 if np.random.rand()<0.3 else 0.15)+0.12*np.abs(n),
    w_fn  =lambda t,i: 0.12+0.25*np.random.rand(),
    phi_fn=lambda t,i,p,n: p+dt*(2.5*n-0.04*p),   # near-random walk
    label ="Psychosis"))

# 6. TRAUMA — F0 spike → F1 surge → F2 rises → collapse
def F0_tr(t,i,n):
    if 4<t<8:   return 4.0+0.5*np.abs(n)
    elif 8<t<18: return 1.8-0.14*(t-8)+0.2*np.abs(n)
    else:        return 0.18+0.04*n
sc.append(simulate(
    F0_fn =F0_tr,
    w_fn  =lambda t,i: 0.04 if t<15 else 0.12,
    phi_fn=lambda t,i,p,n: p+dt*((1.5*np.sin(t*4)*( 1 if 4<t<12 else 0))-0.3*p+0.15*n),
    label ="Trauma"))

COLS=["#1E8449","#8E44AD","#D35400","#2C3E50","#C0392B","#1F618D"]

# ── Figure ────────────────────────────────────────────────────────────────
fig=plt.figure(figsize=(22,26)); fig.patch.set_facecolor("#080C14")
gs=gridspec.GridSpec(5,3,figure=fig,hspace=0.44,wspace=0.30,
                     left=0.06,right=0.97,top=0.94,bottom=0.03)
W="#ECF0F1"
def sax(ax,title):
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors="#BDC3C7",labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#2C3E50")
    ax.set_title(title,color=W,fontsize=9.5,fontweight="bold",pad=7)

# ROW 0 — F1 all six
ax=fig.add_subplot(gs[0,:])
for s,c in zip(sc,COLS):
    ax.plot(t_,s["F1"],color=c,lw=1.8,label=s["label"],alpha=0.92)
ax.axhline(F1_max,color=W,ls="--",lw=0.8,alpha=0.35,label=f"F₁ᵐᵃˣ={F1_max}")
ax.axhline(F1_crit,color="#E74C3C",ls=":",lw=1.0,alpha=0.55,label=f"F₁^crit={F1_crit}")
ax.set_ylabel("F₁",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=8,facecolor="#0D1117",labelcolor=W,framealpha=0.7,ncol=8)
sax(ax,"F₁ Dynamics — Accumulator-Dissolver ODE  [dF₁/dt = α·F₀·(1−F₁/F₁ᵐᵃˣ) − β·F₁·w]")

# ROW 1 & 2 — individual scenario plots
for idx,(s,c) in enumerate(zip(sc,COLS)):
    row=1+(idx//3); col=idx%3
    ax=fig.add_subplot(gs[row,col])
    ax.plot(t_,s["F0"],color="#3498DB",lw=1.1,label="F₀",alpha=0.75)
    ax.plot(t_,s["F1"],color=c,lw=1.7,label="F₁")
    ax2=ax.twinx()
    ax2.plot(t_,s["Cm"],color="#E74C3C",lw=1.5,ls="--",label="C_m^int",alpha=0.9)
    ax2.fill_between(t_,s["Cm"],alpha=0.08,color="#E74C3C")
    ax2.set_ylim(-0.05,1.15)
    ax2.tick_params(colors="#BDC3C7",labelsize=7)
    ax2.set_ylabel("C_m^int",color="#E74C3C",fontsize=7)
    ax.set_xlabel("t",color="#BDC3C7",fontsize=8)
    ax.legend(fontsize=7,facecolor="#0D1117",labelcolor=W,framealpha=0.5)
    sax(ax,f"{s['label']}")

# ROW 3 — v_path OCD vs Depression
ax=fig.add_subplot(gs[3,0])
ax.plot(t_,sc[1]["v"],color=COLS[1],lw=1.8,label="OCD")
ax.plot(t_,sc[3]["v"],color=COLS[3],lw=1.8,label="Depression",ls="--")
ax.axhline(0.05,color=W,ls=":",lw=0.8,alpha=0.5,label="v_min")
ax.set_ylabel("v_path = ‖ds/dt‖",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=8,facecolor="#0D1117",labelcolor=W,framealpha=0.7)
sax(ax,"v_path: OCD (moving rigid) vs Depression (frozen)  [M3 fix]")

# ROW 3 — S_eq
ax=fig.add_subplot(gs[3,1])
for s,c in zip(sc,COLS):
    ax.plot(t_,np.clip(s["S"],0,5),color=c,lw=1.5,label=s["label"],alpha=0.85)
ax.axhline(1.0,color=W,ls="--",lw=1.0,alpha=0.55,label="S_eq=1")
ax.fill_between(t_,0,1,alpha=0.07,color="#E74C3C")
ax.set_ylim(0,4.5)
ax.set_ylabel("S_eq",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=7,facecolor="#0D1117",labelcolor=W,framealpha=0.6,ncol=3)
sax(ax,"Stability S_eq = β·w / (α·F₀/F₁)")

# ROW 3 — Phase portrait F1 vs F2
ax=fig.add_subplot(gs[3,2])
for s,c in zip(sc,COLS):
    ax.scatter(s["F1"][::4],s["F2"][::4],c=c,s=6,alpha=0.35)
    ax.scatter(s["F1"][-1],s["F2"][-1],c=c,s=100,marker="D",
               edgecolors=W,lw=0.8,label=s["label"],zorder=10)
ax.axvline(F1_crit,color="#E74C3C",ls=":",lw=1.0,alpha=0.6)
ax.axhline(F2_crit,color="#E74C3C",ls=":",lw=1.0,alpha=0.6)
ax.set_xlabel("F₁",color="#BDC3C7",fontsize=9)
ax.set_ylabel("F₂",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=8,facecolor="#0D1117",labelcolor=W,framealpha=0.7)
sax(ax,"Phase Portrait (F₁, F₂) — Six States")

# ROW 4 — F2 diagnostic + C_m smooth vs hard
ax=fig.add_subplot(gs[4,0:2])
for s,c in zip(sc,COLS):
    ax.plot(t_,s["F2"],color=c,lw=1.5,label=s["label"],alpha=0.85)
ax.axhline(F2_crit,color="#E74C3C",ls="--",lw=1.0,alpha=0.6,label=f"F₂^crit={F2_crit}")
ax.set_ylabel("F₂ = Var(dφ/dt)",color="#BDC3C7",fontsize=9)
ax.set_xlabel("Time",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=8,facecolor="#0D1117",labelcolor=W,framealpha=0.6,ncol=6)
sax(ax,"F₂ — Conflict Instability (Critical Slowing Down Indicator)")

# ROW 4 — sigma product illustration
ax=fig.add_subplot(gs[4,2])
f1r=np.linspace(0,3.5,200); f2r=np.linspace(0,1.5,200)
F1g,F2g=np.meshgrid(f1r,f2r)
Cmg=sig(F1g,F1_crit,k_sig)*sig(F2g,F2_crit,k_sig)
im=ax.contourf(F1g,F2g,Cmg,levels=25,cmap="RdYlGn_r",alpha=0.9)
plt.colorbar(im,ax=ax,label="C_m^int")
ax.axvline(F1_crit,color=W,ls="--",lw=0.8,alpha=0.7)
ax.axhline(F2_crit,color=W,ls="--",lw=0.8,alpha=0.7)
for s,c,mk in zip(sc,COLS,["o","s","^","D","*","P"]):
    ax.scatter(s["F1"].mean(),s["F2"].mean(),color=c,s=120,
               marker=mk,edgecolors=W,lw=0.8,label=s["label"],zorder=10)
ax.set_xlabel("F₁",color="#BDC3C7",fontsize=9)
ax.set_ylabel("F₂",color="#BDC3C7",fontsize=9)
ax.legend(fontsize=7,facecolor="#0D1117",labelcolor=W,framealpha=0.7,ncol=3)
sax(ax,"C_m^int = σ(F₁)·σ(F₂)  [Smooth joint threshold, M2 fix]")

fig.text(0.5,0.97,
    "UDMM Friction Spectrum (F₀/F₁/F₂)  |  Corrected Dynamic Hierarchy  |  UDMM Research Programme",
    ha="center",va="top",color=W,fontsize=12,fontweight="bold")

plt.savefig("/mnt/user-data/outputs/UDMM_Friction_Spectrum.png",
            dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(); print("Figure saved.")

print("\nSummary (time-averaged):")
print(f"{'State':<12}{'F0':>7}{'F1':>7}{'F2':>7}{'Cm':>7}{'v':>7}{'S':>7}")
print("─"*51)
for s in sc:
    print(f"{s['label']:<12}{s['F0'].mean():>7.3f}{s['F1'].mean():>7.3f}"
          f"{s['F2'].mean():>7.3f}{s['Cm'].mean():>7.3f}"
          f"{s['v'].mean():>7.3f}{np.clip(s['S'],0,5).mean():>7.3f}")
