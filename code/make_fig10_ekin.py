"""Fig 10 (outflow rainback): specific kinetic energy of a parcel launched from R_launch = 1 kpc, versus radius.

A parcel launched with v_init has e_init = Phi(R_launch) + v_init^2 / 2 (conserved), so along its trajectory
    v^2(r) / 2 = e_init - Phi(r),
which drops to zero at R_max (bound) or tends to e_init > 0 (unbound). Potential as in the paper text:
    Phi(r) = -G M_* / r - 4 pi G rho_0 R_s^3 / r * ln(1 + r / R_s)
with M_* = eps_term M_g0 (point mass outside the 500 pc starburst), and an NFW halo of M_200 = 1e11 Msun at z = 10
(Planck18), c = 10, R_s = r_200 / c, rho_0 = rho_c(z) (200/3) c^3 / (ln(1+c) - c/(1+c)). This gives
v_esc(1 kpc) = 663 km/s and reproduces the R_max quoted in the text (1.4 / 2 kpc for 240 / 330 km/s).

Phase bands: median +- 1 sigma over t = 6.7-20 Myr of <v_out> per phase in the fiducial run, from the cluster outflow
rates (data/outflow_rates/outflow_rates_SNeRHD.npy, r = 1 kpc, WNM/WIM split at f_ion = 0.5): these medians are the
values in the text (HIM 1830, WIM 320, WNM 250 km/s).

Usage:  ~/miniconda3/bin/python make_fig10_ekin.py [R_max_axis_kpc (default 20)] [outfile]
Writes  ../../outputs/figures/fig10_ekin.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.integrate import quad
from scipy.optimize import brentq
from astropy.cosmology import Planck18 as cosmo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
import sys
XMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0     # outer edge of the (log) R axis [kpc]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "outputs", "figures", "fig10_ekin.png")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 15})

G = 4.30091e-6                                   # kpc (km/s)^2 / Msun
Z, M200, CONC, EPS, R_LAUNCH = 10, 1e11, 10, 0.76, 1.0
T0, T1 = 6.7, 20.0
UNIT = 1e6                                       # (km/s)^2 per 1e16 cm^2 s^-2

M0 = float(np.load(os.path.join(DATA, "cache", "M0.npy")))          # initial gas mass [Msun]
MSTAR = EPS * M0
rhoc = cosmo.critical_density(Z).to(u.Msun / u.kpc ** 3).value
R200 = (3 * M200 / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
RS = R200 / CONC
RHO0 = rhoc * 200 / 3 * CONC ** 3 / (np.log(1 + CONC) - CONC / (1 + CONC))


def phi(r):
    return -G * MSTAR / r - 4 * np.pi * G * RHO0 * RS ** 3 / r * np.log(1 + r / RS)


def ekin(r, v0):
    """v^2/2 = e_init - Phi(r) [(km/s)^2] for a parcel launched at R_LAUNCH with speed v0; NaN where it is <= 0."""
    e = phi(R_LAUNCH) + 0.5 * v0 ** 2 - phi(r)
    return np.where(e > 0, e, np.nan)


KPC_PER_KMS_MYR = 977.79                        # 1 kpc / (1 km/s) in Myr


def rmax_train(v0):
    """(R_max [kpc], t_rain = 2 t_max [Myr]) for a bound parcel launched at R_LAUNCH with speed v0.
    t_max = int_{R_launch}^{R_max} dr / sqrt(2 (e_init - Phi(r))); with r = R_max - u^2 the 1/sqrt endpoint
    singularity becomes a smooth integrand."""
    e = phi(R_LAUNCH) + 0.5 * v0 ** 2
    if e >= 0:
        return np.inf, np.inf
    rmax = brentq(lambda x: e - phi(x), R_LAUNCH, 1e5)
    f = lambda uu: 2 * uu / np.sqrt(max(2 * (e - phi(rmax - uu ** 2)), 1e-300))
    tmax = quad(f, 0, np.sqrt(rmax - R_LAUNCH), limit=200)[0] * KPC_PER_KMS_MYR
    return rmax, 2 * tmax


# launch-velocity statistics per phase (cluster outflow rates, fiducial run, r = 1 kpc)
cl = np.load(os.path.join(DATA, "outflow_rates", "outflow_rates_SNeRHD.npy"))[:, :, 1, :]
t = np.arange(cl.shape[-1]) / 10.0
win = (t >= T0) & (t <= T1)
stats = {}
for ph, j in dict(HIM=1, WIM=2, WNM=3).items():
    v = cl[0, j, win] * 3.0857e13                 # pc/s -> km/s
    v = v[np.isfinite(v) & (cl[1, j, win] > 0)]
    stats[ph] = (np.median(v), np.std(v))
    print(f"{ph}: median {stats[ph][0]:.0f} km/s, std {stats[ph][1]:.0f} km/s")
v_esc = np.sqrt(-2 * phi(R_LAUNCH))
print(f"v_esc(R_launch) = {v_esc:.0f} km/s; R_200 = {R200:.1f} kpc, R_s = {RS:.2f} kpc")

r = np.logspace(np.log10(R_LAUNCH), np.log10(XMAX), 2000)
xfrac = lambda f: R_LAUNCH * (XMAX / R_LAUNCH) ** f      # position a fraction f across the log axis
fig, ax = plt.subplots(figsize=(6.2, 5.8))
# reference trajectories labelled by launch speed
for v0 in (250, 500, 750, 1000, 1250, 1500, 1750, 2000):   # 100 km/s stays below the y range
    ax.plot(r, ekin(r, v0) / UNIT, color="k", lw=0.8, zorder=3)
    if v0 in (500, 1000, 1500, 2000):
        ax.text(xfrac(0.01), 0.5 * v0 ** 2 / UNIT * 1.06, f"{v0} km/s" if v0 == 2000 else f"{v0}", fontsize=10, va="bottom", zorder=4)
# escape case: e_init = 0
ax.plot(r, -phi(r) / UNIT, color="gray", lw=2, alpha=0.7, zorder=2)
xe = xfrac(0.45)
ax.text(xe, -phi(xe) / UNIT * 0.62, r"$v_{\rm esc}$", color="gray", fontsize=16)
# phase bands: median +- 1 sigma launch speed
style = dict(HIM=("hotpink", "--", 0.45), WIM=("green", "--", 0.35), WNM=("green", "-", 0.5))
for ph in ("HIM", "WIM", "WNM"):
    c, ls, a = style[ph]
    med, sig = stats[ph]
    lo, hi = ekin(r, max(med - sig, 0)) / UNIT, ekin(r, med + sig) / UNIT
    ax.fill_between(r, np.nan_to_num(lo, nan=0), hi, where=np.isfinite(hi), color=c, alpha=a, lw=0, zorder=1)
    ax.plot(r, ekin(r, med) / UNIT, color=c, ls=ls, lw=2.2, zorder=4, label=f"{ph} ({med:.0f} km/s)")
ax.axvline(R200, color="gray", lw=2, alpha=0.6, zorder=2)
ax.text(R200 * 1.03, 0.012, r"$R_{\rm vir}$", color="gray", fontsize=16)
ax.set(xscale="log", yscale="log", xlim=(R_LAUNCH, XMAX), ylim=(1e-2, 3),
       xlabel=r"$R$ [kpc]", ylabel=r"$1/2\ v^2$  [$10^{16}$ cm$^2$ s$^{-2}$]")

from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20])); ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.11)
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=10, frameon=False,
           bbox_to_anchor=(0.57, 0.99), handlelength=2.0, columnspacing=0.9)
fig.savefig(OUT, dpi=250)
print("wrote", OUT)
for ph in ("WNM", "WIM"):
    med = stats[ph][0]; e = phi(R_LAUNCH) + 0.5 * med ** 2
    rr = np.logspace(0, 3, 20000); rmax = rr[np.argmax(e - phi(rr) <= 0)]
    print(f"{ph}: R_max = {rmax:.2f} kpc")
